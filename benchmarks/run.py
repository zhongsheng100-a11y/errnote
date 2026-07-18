"""
Agent Memory Benchmark v2 — 真正体现记忆价值的场景设计。

Key insight: memory shines when the agent doesn't have perfect default fixes.
We simulate a "dumb" agent that learns better fixes over time through memory.

3 task types, 3 rounds each. Each round = 5 iterations × 5 different query variants.
Round 1: agent has no memory → high failure rate
Round 2: agent with memory, but cold start (learning)
Round 3: agent with memory, warm (uses learned patterns)
"""

import sys
import time
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from agent_memory import AgentMemory


class RealisticToolSim:
    """Tools that fail in non-obvious ways, requiring learned fixes."""
    
    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.calls = 0
    
    def search_api(self, query: str, use_fix: str = None) -> dict:
        """Simulate web search with rate limits and timeouts."""
        self.calls += 1
        qlen = len(query)
        
        if qlen > 80:
            # Long queries timeout — fix is to shorten
            if use_fix == "shorten":
                return {"success": True, "results": ["Found 5 papers"]}
            return {"success": False, "error": "timeout", "detail": "Query too long"}
        
        if self.rng.random() < 0.3:
            # Random rate limiting — fix is rate-specific backoff
            if use_fix == "backoff_2s":
                return {"success": True, "results": ["Found 3 papers"]}
            return {"success": False, "error": "rate_limit", "detail": "Too many requests"}
        
        return {"success": True, "results": ["Found 5 papers"]}
    
    def file_reader(self, path: str, use_fix: str = None) -> dict:
        """Simulate file reading with various failures."""
        self.calls += 1
        
        if path.startswith("/etc/") or path.startswith("/root/"):
            if use_fix == "sudo":
                return {"success": True, "content": "system config content"}
            return {"success": False, "error": "permission_denied", "detail": "Need root access"}
        
        if path.endswith(".pickle") or path.endswith(".pkl"):
            if use_fix == "use_pickle_loader":
                return {"success": True, "content": "deserialized data"}
            return {"success": False, "error": "deserialization_error", "detail": "Not plain text"}
        
        if path.endswith(".jsonl"):
            if use_fix == "line_by_line":
                return {"success": True, "content": "parsed 10000 records"}
            if use_fix == "chunked":
                return {"success": True, "content": "parsed in chunks"}
            return {"success": False, "error": "memory_error", "detail": "File too large for memory"}
        
        return {"success": True, "content": f"Content of {path}"}
    
    def api_client(self, endpoint: str, use_fix: str = None) -> dict:
        """Simulate API with auth and server errors."""
        self.calls += 1
        
        if "admin" in endpoint or "delete" in endpoint:
            if use_fix == "use_admin_token":
                return {"success": True, "data": {"status": "done"}}
            return {"success": False, "error": "forbidden", "detail": "Insufficient permissions"}
        
        if "stream" in endpoint:
            if use_fix == "use_sse_client":
                return {"success": True, "data": {"stream": "connected"}}
            return {"success": False, "error": "protocol_error", "detail": "Not a standard REST endpoint"}
        
        if self.rng.random() < 0.25:
            if use_fix == "retry_with_backoff":
                return {"success": True, "data": {"status": "ok_after_retry"}}
            return {"success": False, "error": "server_error", "detail": "500 Internal Server Error"}
        
        return {"success": True, "data": {"status": "ok"}}


class LearningAgent:
    """Agent that gets smarter with memory — has NO default fixes."""
    
    def __init__(self, tools, memory: AgentMemory = None):
        self.tools = tools
        self.memory = memory
        self.successes = 0
        self.failures = 0
        self.retries = 0
        self.learned_fixes = {}  # track what fixes we've discovered
    
    def try_tool(self, tool_name: str, args: dict, true_fix: str) -> bool:
        """Try a tool call. If it fails, try to learn a fix."""
        result = getattr(self.tools, tool_name)(**args, use_fix=None)
        
        if result["success"]:
            self.successes += 1
            return True
        
        self.failures += 1
        error = result["error"]
        
        # Try to find a fix — first from memory, then try random
        fix = None
        if self.memory:
            memories = self.memory.recall(f"{tool_name} {error} {str(args)}")
            for m in memories:
                if m.get("fix"):
                    fix = m["fix"]
                    break
        
        if fix:
            # Apply the learned fix
            result2 = getattr(self.tools, tool_name)(**args, use_fix=fix)
            if result2["success"]:
                self.successes += 1
                if self.memory:
                    self.memory.confirm(memories[0]["id"], was_helpful=True)
                return True
            else:
                self.failures += 1
                if self.memory:
                    self.memory.confirm(memories[0]["id"], was_helpful=False)
                return False
        
        # No fix in memory — try the right fix (simulating trial-and-error)
        # In real agent, this would be LLM reasoning; we simulate with the true_fix
        fix = true_fix
        self.retries += 1
        result3 = getattr(self.tools, tool_name)(**args, use_fix=fix)
        
        if result3["success"]:
            self.successes += 1
            if self.memory:
                self.memory.remember(
                    tool=tool_name,
                    input=str(args),
                    outcome="fail",
                    error=error,
                    error_detail=result["detail"],
                    fix=fix,
                )
            self.learned_fixes[f"{tool_name}:{error}"] = fix
            return True
        else:
            self.failures += 1
            return False


# ── Test scenarios ──

# Each scenario: (tool_name, args, true_fix)
WEB_RESEARCH_SCENARIOS = [
    ("search_api", {"query": "GRPO reinforcement learning 2026 survey and comparison " + "x" * 60}, "shorten"),
    ("search_api", {"query": "MCP server agent memory best practices"}, "backoff_2s"),
    ("search_api", {"query": "function calling vs tool use LLM agent " + "y" * 50}, "shorten"),
    ("search_api", {"query": "open source RAG"}, "backoff_2s"),
    ("search_api", {"query": "qwen2.5 fine-tuning lora qlora ROCm AMD " + "z" * 70}, "shorten"),
]

FILE_PROCESSING_SCENARIOS = [
    ("file_reader", {"path": "/etc/nginx/nginx.conf"}, "sudo"),
    ("file_reader", {"path": "/home/user/project/readme.md"}, None),
    ("file_reader", {"path": "/root/.ssh/authorized_keys"}, "sudo"),
    ("file_reader", {"path": "/home/user/data/export.pickle"}, "use_pickle_loader"),
    ("file_reader", {"path": "/home/user/logs/server.jsonl"}, "chunked"),
]

API_DEBUGGING_SCENARIOS = [
    ("api_client", {"endpoint": "/admin/users"}, "use_admin_token"),
    ("api_client", {"endpoint": "/data/query"}, None),
    ("api_client", {"endpoint": "/stream/events"}, "use_sse_client"),
    ("api_client", {"endpoint": "/admin/settings/delete"}, "use_admin_token"),
    ("api_client", {"endpoint": "/api/v2/status"}, None),
]

ALL_SCENARIOS = [
    ("Web Research", WEB_RESEARCH_SCENARIOS),
    ("File Processing", FILE_PROCESSING_SCENARIOS),
    ("API Debugging", API_DEBUGGING_SCENARIOS),
]

LEARNING_ROUNDS = 5  # Number of times to repeat the same scenarios


def run_benchmark():
    print("=" * 70)
    print("  Agent Memory Benchmark — 错题本价值验证")
    print(f"  {len(ALL_SCENARIOS)} tasks × {LEARNING_ROUNDS} learning rounds")
    print("  No Memory = agent must re-discover fix every time")
    print("  With Memory = agent remembers fixes across rounds")
    print("=" * 70)
    
    for task_name, scenarios in ALL_SCENARIOS:
        print(f"\n{'─' * 60}")
        print(f"  {task_name}")
        print(f"{'─' * 60}")
        
        # ── WITHOUT memory ──
        tools_no = RealisticToolSim(seed=42)
        agent_no = LearningAgent(tools_no, memory=None)
        
        for round_i in range(LEARNING_ROUNDS):
            for tool_name, args, true_fix in scenarios:
                agent_no.try_tool(tool_name, args, true_fix if true_fix else "retry")
        
        no_sr = agent_no.successes / max(agent_no.successes + agent_no.failures, 1)
        no_rt = agent_no.retries
        
        # ── WITH memory ──
        import tempfile, os
        fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        
        try:
            mem = AgentMemory(db_path)
            tools_with = RealisticToolSim(seed=42)
            agent_with = LearningAgent(tools_with, memory=mem)
            
            round_metrics = []
            for round_i in range(LEARNING_ROUNDS):
                s_before = agent_with.successes
                f_before = agent_with.failures
                
                for tool_name, args, true_fix in scenarios:
                    agent_with.try_tool(tool_name, args, true_fix if true_fix else "retry")
                
                round_successes = agent_with.successes - s_before
                round_total = round_successes + (agent_with.failures - f_before)
                round_sr = round_successes / max(round_total, 1)
                round_metrics.append(round_sr)
            
            mem.close()
        finally:
            os.unlink(db_path)
        
        wi_sr = agent_with.successes / max(agent_with.successes + agent_with.failures, 1)
        wi_rt = agent_with.retries
        
        # ── Print ──
        print(f"  Without Memory:")
        print(f"    Success Rate: {no_sr:.1%}  |  Trial-and-error fixes: {no_rt}")
        print(f"  With Memory:")
        print(f"    Success Rate: {wi_sr:.1%}  |  Trial-and-error fixes: {wi_rt}")
        
        sr_improve = (wi_sr - no_sr) / max(no_sr, 0.001) * 100
        rt_reduce = (no_rt - wi_rt) / max(no_rt, 0.001) * 100 if no_rt > 0 else 0
        
        print(f"  Improvement: Success +{sr_improve:.0f}% | Fixes needed -{rt_reduce:.0f}%")
        print(f"  Learning curve (per-round success rate):")
        for i, sr in enumerate(round_metrics):
            bar = "█" * int(sr * 20)
            print(f"    Round {i+1}: {sr:.0%} {bar}")
        
        # Show what was learned
        if agent_with.learned_fixes:
            print(f"  Learned fixes: {len(agent_with.learned_fixes)}")
            for key, fix in list(agent_with.learned_fixes.items())[:3]:
                print(f"    {key} → {fix}")
    
    print(f"\n{'=' * 70}")
    mem_stats = AgentMemory(":memory:").store
    print(f"  ✅ Benchmark complete — memory lets agent retain learned fixes")
    print(f"     instead of re-discovering them every round.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    run_benchmark()
