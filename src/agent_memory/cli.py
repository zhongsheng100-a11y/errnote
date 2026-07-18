"""CLI entry point for agent-memory."""

import sys
import argparse


def main():
    parser = argparse.ArgumentParser(
        description="Agent Memory — the error notebook for AI agent tool calls",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_parser = sub.add_parser("serve", help="Start MCP server")
    serve_parser.add_argument("--db", default="./agent_memory.db", help="SQLite database path")

    # stats
    stats_parser = sub.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--db", default="./agent_memory.db", help="SQLite database path")

    # version
    sub.add_parser("version", help="Show version")

    args = parser.parse_args()

    if args.command == "serve":
        from agent_memory.mcp_server import main as serve_main
        sys.argv = [sys.argv[0], "--db", args.db]
        serve_main()
    elif args.command == "stats":
        from agent_memory import AgentMemory
        mem = AgentMemory(args.db)
        s = mem.stats()
        print(f"Total: {s['total']} | Failures: {s['failures']} | Avg confidence: {s['avg_confidence']:.0%}")
        mem.close()
    elif args.command == "version":
        from agent_memory import __version__
        print(f"agent-memory v{__version__}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
