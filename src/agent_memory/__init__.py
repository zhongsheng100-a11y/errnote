"""
Agent Memory — the error notebook for AI agent tool calls.

Usage:
    from agent_memory import AgentMemory
    mem = AgentMemory("./agent_errors.db")
    mem.remember(tool="web_search", input="...", outcome="fail", error="timeout")
    results = mem.recall("how to search papers")
"""

from .core import AgentMemory

__version__ = "0.1.0"
__all__ = ["AgentMemory"]
