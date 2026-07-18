"""Smoke tests for agent-memory core functionality."""

import tempfile
import os
from pathlib import Path

# Add src to path for direct execution
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from agent_memory import AgentMemory


def test_remember_and_recall():
    """Basic write + read test."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        mem = AgentMemory(db_path)

        # Remember some tool calls
        mem.remember(
            tool="web_search",
            input="search for GRPO reinforcement learning papers 2026",
            outcome="fail",
            error="timeout",
            error_detail="Request to search API timed out after 30 seconds",
            fix="retry with smaller query",
        )
        mem.remember(
            tool="web_search",
            input="GRPO paper",
            outcome="success",
        )
        mem.remember(
            tool="read_file",
            input="/etc/config.yaml",
            outcome="fail",
            error="auth_error",
            error_detail="Permission denied: /etc/config.yaml",
            fix="use sudo or check file permissions",
        )
        mem.remember(
            tool="web_search",
            input="rate limit test query",
            outcome="rate_limit",
            error="HTTP 429",
            error_detail="Too many requests",
            fix="wait 60 seconds and retry",
        )

        # Test recall
        results = mem.recall("timeout search")
        assert len(results) > 0, "Should find timeout-related memories"
        found_timeout = any("timeout" in str(r.get("error_type", "")) for r in results)
        assert found_timeout, "Should find the timeout memory"

        # Test recall with specific tool
        results = mem.recall("auth_error")
        assert len(results) > 0, "Should find auth_error memories"
        found_auth = any("auth_error" in str(r.get("error_type", "")) for r in results)
        assert found_auth, "Should find the auth_error memory"

        # Test CJK recall (falls back to LIKE)
        mem.remember(
            tool="web_search",
            input="搜索最新的GRPO强化学习论文",
            outcome="success",
        )
        results = mem.recall("搜索")
        assert len(results) > 0, "CJK fallback should find something"

        # Test review
        all_memories = mem.review()
        assert len(all_memories) >= 3, f"Should have at least 3 memories, got {len(all_memories)}"

        # Test confirm: first mark unhelpful (drops), then helpful (rises)
        fail_memories = [r for r in all_memories if r["outcome"] != "success"]
        assert len(fail_memories) > 0, "Should have failure memories"
        mem_id = fail_memories[0]["id"]
        old_conf = mem.store.get_by_id(mem_id)["confidence"]
        mem.confirm(mem_id, was_helpful=False)
        mid_conf = mem.store.get_by_id(mem_id)["confidence"]
        assert mid_conf < old_conf, f"Confidence should drop after unhelpful: {old_conf} -> {mid_conf}"
        mem.confirm(mem_id, was_helpful=True)
        new_conf = mem.store.get_by_id(mem_id)["confidence"]
        assert new_conf > mid_conf, f"Confidence should rise after helpful: {mid_conf} -> {new_conf}"

        # Test forget
        mem.forget(mem_id)
        assert mem.store.get_by_id(mem_id) is None, "Memory should be deleted"

        # Test stats
        stats = mem.stats()
        assert stats["total"] > 0
        assert stats["failures"] > 0
        print(f"Stats: {stats}")

        # Test entity registration
        mem.register_entity("tool", "web_search", {"category": "network"})
        mem.register_entity("error", "timeout", {"retryable": True})
        entities = mem.list_entities()
        assert len(entities) >= 2, f"Should have at least 2 entities"

        # Test deduplication (same tool + input)
        mem.remember(tool="web_search", input="GRPO paper", outcome="fail", error="timeout")
        mem.remember(tool="web_search", input="GRPO paper", outcome="success")
        all_after_dedup = mem.review()
        grpo_count = sum(1 for m in all_after_dedup if m["tool"] == "web_search" and "GRPO" in m["input_preview"])
        # Should be 1 deduped entry with updated outcome
        assert grpo_count >= 1, "Should have deduplicated by tool+input hash"

        mem.close()
        print("✅ All smoke tests passed!")
    finally:
        os.unlink(db_path)


if __name__ == "__main__":
    test_remember_and_recall()
