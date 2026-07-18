"""
MCP Server for Agent Memory — exposes remember/recall/review/forget as MCP tools.

Usage:
    python -m agent_memory.mcp_server --db ./agent_errors.db
    # or: agent-memory serve --db ./agent_errors.db
"""

import sys
import os
import argparse
from pathlib import Path

# Ensure the package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_memory import AgentMemory

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    print("MCP not installed. Run: pip install agent-memory[mcp]", file=sys.stderr)
    sys.exit(1)

# Global memory instance
_memory: AgentMemory = None


def get_memory() -> AgentMemory:
    global _memory
    if _memory is None:
        db_path = os.environ.get("AGENT_MEMORY_DB", "./agent_memory.db")
        _memory = AgentMemory(db_path)
    return _memory


# ── MCP Server ──

mcp = FastMCP("Agent Memory — 错题本")


@mcp.tool()
def remember(
    tool: str,
    input_text: str,
    outcome: str,
    error: str = "",
    error_detail: str = "",
    fix: str = "",
) -> str:
    """
    Log a tool call outcome to the agent's error notebook.

    Args:
        tool: Tool name (e.g., 'web_search', 'read_file', 'api_call')
        input_text: The input/query passed to the tool
        outcome: One of 'success', 'fail', 'timeout', 'rate_limit', 'auth_error'
        error: Short error label (e.g., 'TimeoutError')
        error_detail: Detailed error message
        fix: What fixed it (if retried successfully)

    Returns:
        Confirmation with memory ID.
    """
    mem = get_memory()
    mid = mem.remember(
        tool=tool,
        input=input_text,
        outcome=outcome,
        error=error,
        error_detail=error_detail,
        fix=fix,
    )
    return f"✅ Memory #{mid} saved: {tool} → {outcome}" + (f" (fix: {fix})" if fix else "")


@mcp.tool()
def recall(query: str, top_k: int = 5) -> str:
    """
    Search past tool call experiences for relevant lessons.

    Args:
        query: What you want to learn from past experiences
        top_k: Max results (default 5)

    Returns:
        Formatted list of relevant memories with confidence scores.
    """
    mem = get_memory()
    results = mem.recall(query, top_k=top_k)

    if not results:
        return "📭 No relevant memories found. This is a new experience."

    lines = [f"📚 Found {len(results)} relevant memories:"]
    for i, r in enumerate(results):
        conf = "🟢" if r["confidence"] >= 0.7 else "🟡" if r["confidence"] >= 0.4 else "🔴"
        lines.append(
            f"\n{i+1}. [{r['outcome']}] {r['tool']} {conf} conf={r['confidence']:.0%}"
        )
        if r["error_type"]:
            lines.append(f"   Error: {r['error_type']}")
        if r["fix"]:
            lines.append(f"   Fix: {r['fix']}")
        lines.append(f"   Context: {r['input_preview'][:100]}")

    return "\n".join(lines)


@mcp.tool()
def review(
    limit: int = 20,
    offset: int = 0,
    outcome_filter: str = "",
    low_confidence_only: bool = False,
) -> str:
    """
    List memories for human inspection and correction.

    Args:
        limit: Max results (default 20)
        offset: Pagination offset
        outcome_filter: Filter by outcome ('fail', 'timeout', etc.)
        low_confidence_only: Only show low-confidence (<0.5) memories

    Returns:
        Table of memories.
    """
    mem = get_memory()
    results = mem.review(
        limit=limit,
        offset=offset,
        outcome_filter=outcome_filter or None,
        low_confidence_only=low_confidence_only,
    )

    if not results:
        return "📭 No memories found."

    lines = [f"📋 Memories ({len(results)} shown):"]
    for r in results:
        conf = "🟢" if r["confidence"] >= 0.7 else "🟡" if r["confidence"] >= 0.4 else "🔴"
        outcome_icon = "✅" if r["outcome"] == "success" else "❌"
        lines.append(
            f"  #{r['id']} {outcome_icon} {r['tool']} {conf} c={r['confidence']:.0%} "
            f"uses={r['uses']}"
        )
        if r["error_type"]:
            lines.append(f"       error={r['error_type']}")
        if r["fix"]:
            lines.append(f"       fix={r['fix']}")

    return "\n".join(lines)


@mcp.tool()
def forget(memory_id: int) -> str:
    """
    Remove a memory by ID.

    Args:
        memory_id: The memory ID to remove (from review output)

    Returns:
        Confirmation.
    """
    mem = get_memory()
    mem.forget(memory_id)
    return f"🗑️ Memory #{memory_id} removed."


@mcp.tool()
def confirm(memory_id: int, was_helpful: bool) -> str:
    """
    After using a memory, confirm whether it helped.
    Helpful → +0.2 confidence. Unhelpful → -0.2 confidence.

    Args:
        memory_id: The memory ID
        was_helpful: True if the memory helped, False if it misled

    Returns:
        Updated confidence.
    """
    mem = get_memory()
    mem.confirm(memory_id, was_helpful)
    updated = mem.store.get_by_id(memory_id)
    if updated:
        return f"{'👍' if was_helpful else '👎'} Memory #{memory_id} confidence now {updated['confidence']:.0%}"
    return f"Memory #{memory_id} not found."


@mcp.tool()
def stats() -> str:
    """
    Show memory statistics.

    Returns:
        Total memories, failure rate, average confidence.
    """
    mem = get_memory()
    s = mem.stats()
    return (
        f"📊 Agent Memory Stats\n"
        f"  Total memories: {s['total']}\n"
        f"  Failures: {s['failures']}\n"
        f"  Avg confidence: {s['avg_confidence']:.0%}"
    )


def main():
    parser = argparse.ArgumentParser(description="Agent Memory MCP Server")
    parser.add_argument("--db", default="./agent_memory.db", help="SQLite database path")
    args = parser.parse_args()

    os.environ["AGENT_MEMORY_DB"] = args.db
    print(f"🧠 Agent Memory MCP Server starting (db: {args.db})", file=sys.stderr)
    mcp.run()


if __name__ == "__main__":
    main()
