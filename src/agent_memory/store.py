"""
SQLite storage with FTS5 trigram tokenizer for tool call memory.

Schema:
    memories: tool call records with outcome, error_type, fix, confidence
    entities: known tools, error types, environments
"""

import sqlite3
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tool TEXT NOT NULL,
    input_digest TEXT NOT NULL,
    input_preview TEXT NOT NULL DEFAULT '',
    outcome TEXT NOT NULL CHECK(outcome IN ('success','fail','timeout','rate_limit','auth_error','other')),
    error_type TEXT DEFAULT '',
    error_detail TEXT DEFAULT '',
    fix TEXT DEFAULT '',
    confidence REAL NOT NULL DEFAULT 1.0,
    uses INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    last_used_at REAL NOT NULL,
    env_fingerprint TEXT DEFAULT '',
    UNIQUE(tool, input_digest)
);

CREATE TABLE IF NOT EXISTS entities (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,
    name TEXT NOT NULL,
    properties_json TEXT DEFAULT '{}',
    first_seen_at REAL NOT NULL,
    UNIQUE(type, name)
);

CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    tool,
    input_preview,
    error_type,
    error_detail,
    fix,
    content='memories',
    content_rowid='id',
    tokenize='trigram'
);

-- Triggers to keep FTS index in sync
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, tool, input_preview, error_type, error_detail, fix)
    VALUES (new.id, new.tool, new.input_preview, new.error_type, new.error_detail, new.fix);
END;

CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, tool, input_preview, error_type, error_detail, fix)
    VALUES ('delete', old.id, old.tool, old.input_preview, old.error_type, old.error_detail, old.fix);
END;

CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, tool, input_preview, error_type, error_detail, fix)
    VALUES ('delete', old.id, old.tool, old.input_preview, old.error_type, old.error_detail, old.fix);
    INSERT INTO memories_fts(rowid, tool, input_preview, error_type, error_detail, fix)
    VALUES (new.id, new.tool, new.input_preview, new.error_type, new.error_detail, new.fix);
END;

CREATE INDEX IF NOT EXISTS idx_memories_tool ON memories(tool);
CREATE INDEX IF NOT EXISTS idx_memories_outcome ON memories(outcome);
CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
CREATE INDEX IF NOT EXISTS idx_memories_created ON memories(created_at);
"""


def _hash(text: str) -> str:
    """Deterministic hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _truncate(text: str, max_chars: int = 500) -> str:
    """Truncate text, keeping head and tail."""
    if len(text) <= max_chars:
        return text
    head = max_chars // 2
    tail = max_chars - head - 20
    return text[:head] + f"\n... [{len(text) - max_chars} chars omitted] ...\n" + text[-tail:]


class MemoryStore:
    """SQLite-backed storage for tool call memories."""

    def __init__(self, db_path: str | Path, max_memories: int = 10000):
        self.db_path = Path(db_path)
        self.max_memories = max_memories
        self._conn: Optional[sqlite3.Connection] = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(SCHEMA_SQL)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # ── Write operations ──

    def insert_memory(
        self,
        tool: str,
        input_text: str,
        outcome: str,
        error_type: str = "",
        error_detail: str = "",
        fix: str = "",
        env_fingerprint: str = "",
    ) -> int:
        """Insert or update a tool call memory. Returns row id."""
        now = time.time()
        digest = _hash(input_text)
        preview = _truncate(input_text)

        conn = self.conn
        row = conn.execute(
            "SELECT id, uses, confidence FROM memories WHERE tool=? AND input_digest=?",
            (tool, digest),
        ).fetchone()

        if row:
            # Update existing: bump uses, adjust confidence
            row_id, old_uses, old_conf = row
            # Success after previous failure = strong positive signal
            if outcome == "success":
                new_conf = min(1.0, old_conf + 0.2)
            else:
                new_conf = max(0.1, old_conf - 0.1)
            conn.execute(
                "UPDATE memories SET outcome=?, error_type=?, error_detail=?, fix=?, "
                "confidence=?, uses=?, last_used_at=?, env_fingerprint=? WHERE id=?",
                (outcome, error_type, error_detail, fix, new_conf, old_uses + 1, now, env_fingerprint, row_id),
            )
            return row_id
        else:
            # New memory
            cursor = conn.execute(
                "INSERT INTO memories(tool, input_digest, input_preview, outcome, error_type, error_detail, fix, "
                "confidence, uses, created_at, last_used_at, env_fingerprint) "
                "VALUES (?,?,?,?,?,?,?,?,1,?,?,?)",
                (tool, digest, preview, outcome, error_type, error_detail, fix, 1.0, now, now, env_fingerprint),
            )
            self._maybe_evict()
            return cursor.lastrowid

    def _maybe_evict(self):
        """Evict oldest low-confidence memories if over capacity."""
        conn = self.conn
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if count > self.max_memories:
            excess = count - self.max_memories
            conn.execute(
                "DELETE FROM memories WHERE id IN ("
                "  SELECT id FROM memories ORDER BY confidence ASC, last_used_at ASC LIMIT ?"
                ")",
                (excess,),
            )

    # ── Read operations ──

    def _sanitize_fts5(self, query: str) -> str:
        """Remove characters that break FTS5 MATCH syntax."""
        # FTS5 treats most punctuation as operators; keep alphanumeric, CJK, spaces
        import re
        clean = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if not clean:
            return "unknown"
        return clean

    def search_fts(
        self,
        query: str,
        top_k: int = 5,
        half_life_days: int = 30,
    ) -> list[dict]:
        """Full-text search with time decay ranking."""
        now = time.time()
        half_life_sec = half_life_days * 86400

        conn = self.conn
        clean_query = self._sanitize_fts5(query)
        # Use FTS5 for primary search, fall back to LIKE for short queries
        rows = conn.execute(
            "SELECT m.id, m.tool, m.input_preview, m.outcome, m.error_type, "
            "m.error_detail, m.fix, m.confidence, m.uses, m.created_at, m.last_used_at, "
            "m.env_fingerprint "
            "FROM memories m "
            "JOIN memories_fts f ON m.id = f.rowid "
            "WHERE memories_fts MATCH ? "
            "ORDER BY rank "
            "LIMIT ?",
            (clean_query, top_k * 2),
        ).fetchall()

        # If FTS returned nothing (short CJK queries), try LIKE fallback
        if not rows:
            like_pattern = f"%{query}%"
            rows = conn.execute(
                "SELECT id, tool, input_preview, outcome, error_type, "
                "error_detail, fix, confidence, uses, created_at, last_used_at, "
                "env_fingerprint "
                "FROM memories "
                "WHERE tool LIKE ? OR input_preview LIKE ? OR error_type LIKE ? OR fix LIKE ? "
                "ORDER BY confidence DESC, last_used_at DESC "
                "LIMIT ?",
                (like_pattern, like_pattern, like_pattern, like_pattern, top_k * 2),
            ).fetchall()

        # Apply time decay and re-rank
        scored = []
        for r in rows:
            age_sec = now - r[9]  # created_at
            decay = 0.5 ** (age_sec / half_life_sec)
            score = r[7] * decay  # confidence * decay
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self._row_to_dict(r, s) for s, r in scored[:top_k]]

    def get_by_id(self, memory_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT id, tool, input_preview, outcome, error_type, "
            "error_detail, fix, confidence, uses, created_at, last_used_at, "
            "env_fingerprint FROM memories WHERE id=?",
            (memory_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row, confidence=None)

    def list_all(self, limit: int = 50, offset: int = 0) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, tool, input_preview, outcome, error_type, "
            "error_detail, fix, confidence, uses, created_at, last_used_at, "
            "env_fingerprint FROM memories ORDER BY last_used_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [self._row_to_dict(r, None) for r in rows]

    def update_confidence(self, memory_id: int, delta: float):
        """Adjust confidence up/down after a memory was used."""
        self.conn.execute(
            "UPDATE memories SET confidence=MAX(0.0, MIN(1.0, confidence+?)) WHERE id=?",
            (delta, memory_id),
        )

    def delete(self, memory_id: int):
        self.conn.execute("DELETE FROM memories WHERE id=?", (memory_id,))

    def stats(self) -> dict:
        conn = self.conn
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        fails = conn.execute("SELECT COUNT(*) FROM memories WHERE outcome!='success'").fetchone()[0]
        avg_conf = conn.execute("SELECT AVG(confidence) FROM memories").fetchone()[0] or 0.0
        return {"total": total, "failures": fails, "avg_confidence": round(avg_conf, 3)}

    # ── Entity operations ──

    def register_entity(self, entity_type: str, name: str, properties: dict = None):
        conn = self.conn
        props_json = json.dumps(properties or {}, ensure_ascii=False)
        conn.execute(
            "INSERT OR IGNORE INTO entities(type, name, properties_json, first_seen_at) VALUES (?,?,?,?)",
            (entity_type, name, props_json, time.time()),
        )

    def list_entities(self, entity_type: str = None) -> list[dict]:
        if entity_type:
            rows = self.conn.execute(
                "SELECT type, name, properties_json, first_seen_at FROM entities WHERE type=? ORDER BY name",
                (entity_type,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT type, name, properties_json, first_seen_at FROM entities ORDER BY type, name"
            ).fetchall()
        return [
            {"type": r[0], "name": r[1], "properties": json.loads(r[2]), "first_seen_at": r[3]}
            for r in rows
        ]

    # ── Helpers ──

    @staticmethod
    def _row_to_dict(row, confidence) -> dict:
        return {
            "id": row[0],
            "tool": row[1],
            "input_preview": row[2],
            "outcome": row[3],
            "error_type": row[4],
            "error_detail": row[5],
            "fix": row[6],
            "confidence": round(row[7], 3) if confidence is None else round(confidence, 3),
            "uses": row[8],
            "created_at": row[9],
            "last_used_at": row[10],
            "env_fingerprint": row[11],
        }
