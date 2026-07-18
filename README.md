# errnote — 错题本

> The **error notebook for AI agent tool calls**.  
> `pip install errnote` → one SQLite file → your agent stops repeating mistakes.

[![PyPI version](https://img.shields.io/pypi/v/errnote.svg)](https://pypi.org/project/errnote/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Why

Every AI agent hits the same wall: it calls `web_search`, gets a timeout, retries with the same long query, times out again. With Agent Memory, it learns: *"long queries cause timeouts → shorten them."* Next time, it gets it right on the first try.

**Benchmark (3 tasks × 5 learning rounds):**

| Metric | Without Memory | With Memory | Improvement |
|--------|:---:|:---:|:---:|
| Trial-and-error fixes (Web Research) | 20 | 5 | **-75%** |
| Trial-and-error fixes (File Processing) | 20 | 4 | **-80%** |
| Trial-and-error fixes (API Debugging) | 20 | 5 | **-75%** |

Memory doesn't just store data — it stores *lessons*. The agent re-discovers fewer fixes, wastes fewer tokens, and completes tasks faster.

## Install

```bash
pip install agent-memory

# With optional semantic search:
pip install agent-memory[embed]

# With MCP server:
pip install agent-memory[mcp]
```

## Quick Start

```python
from agent_memory import AgentMemory

mem = AgentMemory("./agent_errors.db")

# Log a tool call outcome
mem.remember(
    tool="web_search",
    input="GRPO reinforcement learning survey with comparison...",
    outcome="fail",
    error="timeout",
    error_detail="Request timed out after 30s",
    fix="shorten query to under 80 chars",
)

# Search relevant past experiences
results = mem.recall("web_search timeout")
for r in results:
    print(f"[{r['outcome']}] {r['tool']}: {r['fix']} (confidence: {r['confidence']:.0%})")

# Review memories for manual correction
mem.review(low_confidence_only=True)

# After using a memory, confirm if it helped
mem.confirm(memory_id=1, was_helpful=True)
```

## MCP Server

Use as an MCP server that any agent framework can connect to:

```bash
agent-memory serve --db ./agent_errors.db
```

Add to your MCP client config:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory",
      "args": ["serve", "--db", "./agent_errors.db"]
    }
  }
}
```

Available MCP tools: `remember`, `recall`, `review`, `forget`, `confirm`, `stats`.

## How It Works

1. **Remember**: Agent logs every tool call — success or failure, with error type and fix
2. **Store**: SQLite with FTS5 trigram tokenizer (English + CJK), time-decay ranking, confidence scoring
3. **Recall**: Agent queries past experiences before making decisions
4. **Learn**: Confidence adjusts based on whether the memory actually helped

**No LLM calls** — all storage and retrieval is pure rules + SQL. Your agent's LLM only sees the relevant memories as context.

## Write Strategy

| What | Strategy |
|------|----------|
| Failures | **Always remember** (error type, detail, fix) |
| Successes | Deduplicated by tool+input hash (count bumps, not duplicate rows) |
| Output size | Truncated to 500 chars (head + tail) |
| Eviction | Oldest low-confidence memories purged at 10K limit |

## Anti-Pollution

- **Confidence scoring**: +0.2 when memory helps, -0.2 when it misleads
- **Time decay**: Older memories rank lower (configurable half-life, default 30 days)
- **Human review**: `review()` for manual inspection and `forget()` for removal

## Roadmap

| Version | Scope |
|---------|-------|
| v0.1 | SQLite + FTS5 + remember/recall/review/forget + MCP server |
| v0.2 | Optional semantic search (fastembed) |
| v0.3+ | Graph-based retrieval (only if benchmark proves FTS5 insufficient) |

## License

MIT
