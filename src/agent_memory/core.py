"""
AgentMemory — the error notebook for AI agent tool calls.

Core API:
    remember() — log a tool call outcome
    recall()   — search relevant past experiences
    review()   — list memories for human inspection
    forget()   — remove a memory
"""

import time
from pathlib import Path
from typing import Optional

from .store import MemoryStore, _hash, _truncate


# ── Outcome normalization ──
OUTCOME_CLASSIFIER = {
    "success": "success",
    "fail": "fail",
    "failure": "fail",
    "error": "fail",
    "timeout": "timeout",
    "timed_out": "timeout",
    "rate_limit": "rate_limit",
    "rate_limited": "rate_limit",
    "429": "rate_limit",
    "auth_error": "auth_error",
    "unauthorized": "auth_error",
    "401": "auth_error",
    "403": "auth_error",
}


def _normalize_outcome(raw: str) -> str:
    """Normalize user-provided outcome strings to standard categories."""
    return OUTCOME_CLASSIFIER.get(raw.lower().strip(), "other")


class AgentMemory:
    """
    The error notebook for AI agent tool calls.

    Usage:
        mem = AgentMemory("./agent_errors.db")
        mem.remember(tool="web_search", input="GRPO paper", outcome="fail", error="timeout")
        results = mem.recall("web_search timeout")
        mem.review()  # inspect memories
    """

    def __init__(
        self,
        db_path: str | Path = "./agent_memory.db",
        max_memories: int = 10000,
        half_life_days: int = 30,
        env_fingerprint: str = "",
    ):
        self.store = MemoryStore(str(db_path), max_memories=max_memories)
        self.half_life_days = half_life_days
        self.env_fingerprint = env_fingerprint

    # ── Core API ──

    def remember(
        self,
        tool: str,
        input: str,
        outcome: str,
        error: str = "",
        error_detail: str = "",
        fix: str = "",
    ) -> int:
        """
        Log a tool call outcome.

        Args:
            tool: Tool name (e.g., 'web_search', 'read_file')
            input: The input/query passed to the tool
            outcome: One of 'success', 'fail', 'timeout', 'rate_limit', 'auth_error', 'other'
            error: Short error type label (e.g., 'TimeoutError', 'HTTP 429')
            error_detail: Detailed error message (truncated to 500 chars)
            fix: What fixed it (if retried successfully)

        Returns:
            Memory ID (int)
        """
        normalized = _normalize_outcome(outcome)
        return self.store.insert_memory(
            tool=tool,
            input_text=input,
            outcome=normalized,
            error_type=error[:200],
            error_detail=_truncate(error_detail, 500),
            fix=fix[:500],
            env_fingerprint=self.env_fingerprint,
        )

    def recall(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search relevant past experiences.

        Args:
            query: Natural language or keyword query
            top_k: Max results to return

        Returns:
            List of memory dicts, ranked by relevance (confidence * time decay)
        """
        return self.store.search_fts(query, top_k=top_k, half_life_days=self.half_life_days)

    def review(
        self,
        limit: int = 20,
        offset: int = 0,
        low_confidence_only: bool = False,
        outcome_filter: str = None,
    ) -> list[dict]:
        """
        List memories for human inspection.

        Args:
            limit: Max results
            offset: Pagination offset
            low_confidence_only: Only show memories with confidence < 0.5
            outcome_filter: Only show specific outcome type

        Returns:
            List of memory dicts
        """
        results = self.store.list_all(limit=limit, offset=offset)
        if low_confidence_only:
            results = [r for r in results if r["confidence"] < 0.5]
        if outcome_filter:
            results = [r for r in results if r["outcome"] == outcome_filter]
        return results

    def forget(self, memory_id: int):
        """Remove a memory by ID."""
        self.store.delete(memory_id)

    def confirm(self, memory_id: int, was_helpful: bool):
        """
        After using a memory, confirm if it helped.
        Helpful → +0.2 confidence. Unhelpful → -0.2 confidence.
        """
        delta = 0.2 if was_helpful else -0.2
        self.store.update_confidence(memory_id, delta)

    # ── Entity management ──

    def register_entity(self, entity_type: str, name: str, properties: dict = None):
        """Register a known tool, error type, or environment."""
        self.store.register_entity(entity_type, name, properties)

    def list_entities(self, entity_type: str = None) -> list[dict]:
        return self.store.list_entities(entity_type)

    # ── Stats ──

    def stats(self) -> dict:
        return self.store.stats()

    def close(self):
        self.store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
