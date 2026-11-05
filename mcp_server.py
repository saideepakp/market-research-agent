"""
MCP server exposing the market data tools.

Any MCP client can use these tools: this project's LangGraph agent, Claude Desktop,
Claude Code, or another agent.

Run it:
    python mcp_server.py            # HTTP server at http://127.0.0.1:8000/mcp  (for the agent)
    python mcp_server.py --stdio    # stdio mode (for Claude Desktop / Claude Code configs)

The tool functions themselves live in tools.py, so there is one copy of the
data-fetching code. Here we only register them with the MCP server.
"""

import argparse

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv()  # API keys are read by the server, not the agent

from tools import ALL_TOOLS  # noqa: E402  (load .env before tools read env vars)

mcp = FastMCP(
    "market-data",
    instructions="Market data tools: stocks, ETFs, crypto, news, SEC filings, macro data.",
)

# Register each tool: same function, name and docstring the agent saw before.
for t in ALL_TOOLS:
    mcp.add_tool(t.func, name=t.name, description=t.description)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Market data MCP server")
    parser.add_argument("--stdio", action="store_true", help="use stdio transport")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.stdio:
        mcp.run(transport="stdio")
    else:
        mcp.settings.port = args.port
        print(f"Market data MCP server: http://127.0.0.1:{args.port}/mcp")
        mcp.run(transport="streamable-http")
