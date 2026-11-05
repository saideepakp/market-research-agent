"""
Load tools from MCP servers and make them usable by the LangGraph agent.

Servers are listed in mcp_servers.json. To add a third-party MCP server,
add another entry there; its tools are picked up automatically, e.g.

    "some-server": {"transport": "stdio", "command": "npx", "args": ["-y", "some-mcp-server"]}
"""

import asyncio
import json
import os

from langchain_core.tools import BaseTool, StructuredTool
from langchain_mcp_adapters.client import MultiServerMCPClient

CONFIG_PATH = os.getenv("MCP_CONFIG", "mcp_servers.json")


def _make_sync(tool: BaseTool) -> BaseTool:
    """MCP tools are async-only. This app runs the graph synchronously (simpler with
    Streamlit and SqliteSaver), so give each tool a sync version that runs the
    async call to completion. Async callers still use the original coroutine."""

    def run_sync(**kwargs):
        return asyncio.run(tool.ainvoke(kwargs))

    async def run_async(**kwargs):
        return await tool.ainvoke(kwargs)

    return StructuredTool(
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
        func=run_sync,
        coroutine=run_async,
    )


def load_mcp_tools(config_path: str = CONFIG_PATH) -> list[BaseTool]:
    """Connect to every configured MCP server and return all their tools."""
    with open(config_path) as f:
        connections = json.load(f)
    client = MultiServerMCPClient(connections)
    try:
        tools = asyncio.run(client.get_tools())
    except Exception as e:
        raise RuntimeError(
            f"Could not load tools from MCP servers in {config_path}. "
            f"Is the server running? Start it with:  python mcp_server.py\n({e})"
        ) from e
    return [_make_sync(t) for t in tools]
