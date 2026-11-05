# Market Research Agent

A chat agent that researches stocks, ETFs, crypto and the macro economy using
Claude + LangGraph, with a Streamlit UI, SQLite chat memory and Langfuse tracing.
Its data tools are served over **MCP** (Model Context Protocol), so the same tools
also work in Claude Desktop, Claude Code or any other MCP client.

## Architecture

```
 Streamlit UI (app.py)
        │
 LangGraph agent (agent.py) ── Claude API
        │   MCP client (mcp_tools.py, reads mcp_servers.json)
        ▼
 MCP server (mcp_server.py)  ── yfinance, CoinGecko, Finnhub, SEC EDGAR, FRED, DuckDuckGo
        ▲                       (tool code in tools.py, usage limits in usage.py)
 Claude Desktop / Claude Code / other agents can connect here too
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then fill in your keys
```

Only `ANTHROPIC_API_KEY` is required. Web search (DuckDuckGo), stock data (yfinance)
and SEC filings need no key; tools whose key is missing tell the agent to use another source.

## Run (two terminals)

```bash
# Terminal 1: the MCP tool server
python mcp_server.py               # http://127.0.0.1:8000/mcp

# Terminal 2: the chat app
streamlit run app.py               # http://localhost:8501
# or a terminal chat:  python agent.py
```

To skip the server and call tools directly (the original setup), set
`TOOL_MODE=direct` in `.env` and just run `streamlit run app.py`.

## Files

| File | What it does |
|---|---|
| `tools.py`          | Stock, ETF, crypto, news, filings, macro and search tools |
| `derivatives.py`    | Options, futures, FX, volatility and COT tools |
| `common.py`         | Small helpers shared by the two tool files |
| `usage.py`          | Counts every API request and enforces per-source limits |
| `mcp_server.py`     | MCP server that publishes those tools (HTTP or stdio) |
| `mcp_servers.json`  | Which MCP servers the agent connects to |
| `mcp_tools.py`      | MCP client: loads tools from those servers for LangGraph |
| `agent.py`          | LangGraph graph: agent node ↔ tool node loop, SQLite memory, Langfuse |
| `app.py`            | Streamlit chat UI with saved chats in the sidebar |
| `memory.db`         | Created on first run; holds all your chats |
| `usage.db`          | Created on first run; request counts for the usage limits |

## Usage limits

Every request to an outside data source is counted and capped, so you never hit a
provider's own limit. Every source has a monthly cap: below the provider's monthly
limit where one exists, and as a safety budget where it doesn't.

| Source | Per minute | Per day | Per month | Protects |
|---|---|---|---|---|
| CoinGecko  | 30 (5 without key) | 400 | 9,000 | Demo plan: 10,000/month, 100/min |
| Finnhub    | 50  | 2,000 | 30,000 | Free plan: 60/min |
| FRED       | 100 | 2,000 | 20,000 | 120/min |
| SEC EDGAR  | 120 | 1,000 | 10,000 | Fair-access rule: 10/second |
| CFTC       | 30  | 300   | 3,000  | Public API throttling of keyless use |
| yfinance   | 60  | 2,000 | 30,000 | Yahoo throttling (no published limit) |
| DuckDuckGo | 10  | 300   | 6,000  | Search-engine throttling (no published limit) |

One tool call can use several requests: an options overview of 4 expirations uses
5–6 yfinance requests, and a market snapshot uses one per instrument.

When a cap is reached, the tool returns a message telling the agent which other
source to use, and the agent mentions it in its answer. Current usage is shown in
the app sidebar under **API usage**. Change any limit in `.env`, e.g.
`LIMIT_COINGECKO_MONTH=5000` (see `.env.example`). Day and month counts reset at
midnight UTC and on the 1st of the month.

## Derivatives and trading instruments

All in `derivatives.py`; every source is permanently free with no key.

| Tool | What it covers | Source |
|---|---|---|
| `get_options_chain` | One expiration: calls and puts near the money with bid/ask, volume, open interest, IV and Greeks; plus expected move, put/call ratios, max pain, 25-delta skew | Yahoo Finance (delayed ~15 min) + local math |
| `get_options_overview` | Several expirations: IV term structure, expected moves, put/call ratios | Yahoo Finance + local math |
| `option_calculator` | Black-Scholes price, Greeks, probability ITM and breakeven for any inputs | Local math, no API |
| `get_market_snapshot` | Futures (index, energy, metals, agriculture, rates, currencies), FX pairs, Treasury yields, volatility indices (VIX term structure) | Yahoo Finance |
| `get_cot_report` | CFTC Commitments of Traders: speculator vs hedger positioning in 37 futures markets, with a 1-year range | CFTC public API (government data) |

Greeks use the 13-week T-bill yield as the risk-free rate (fallback: `RISK_FREE_RATE`
in `.env`, default 0.04).

## Adding tools

- **Your own tool:** write a function in `tools.py` with `@tool` and a clear docstring
  (Claude reads it to decide when to use the tool), add it to `ALL_TOOLS`, restart the server.
- **A third-party MCP server:** add an entry to `mcp_servers.json`, e.g.
  ```json
  "other-server": {"transport": "stdio", "command": "npx", "args": ["-y", "package-name"]}
  ```
  or `{"transport": "streamable_http", "url": "https://..."}`. Its tools load automatically.
  Only add servers you trust: they run code and return text that Claude will read.

## Use the tools in Claude Desktop

Add this to Claude Desktop's MCP config (Settings → Developer → Edit Config),
using absolute paths on your machine, then restart Claude Desktop:

```json
{
  "mcpServers": {
    "market-data": {
      "command": "/path/to/market-research-agent/.venv/bin/python",
      "args": ["/path/to/market-research-agent/mcp_server.py", "--stdio"]
    }
  }
}
```

## Observability with Langfuse

Create a free account at https://cloud.langfuse.com, make a project, and copy the
keys into `.env`. Every question then shows up as a trace with each Claude call,
tool call, token count and cost, grouped by chat session.

## Notes

- yfinance is an unofficial Yahoo Finance scraper: free, but it can break or rate-limit.
  `pip install -U yfinance` usually fixes it.
- Free-tier limits change; check each provider's pricing page.
- This is a research tool, not investment advice.
