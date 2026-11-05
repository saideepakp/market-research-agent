r"""
The LangGraph agent.

Graph:   START -> agent --(Claude asked for tools?)--> tools -> agent -> ... -> END
                       \--(no tool calls: final answer)--> END

Memory:  SqliteSaver checkpoints the conversation state after every step,
         keyed by thread_id (one thread = one chat). Stored in memory.db.
"""

import os
import sqlite3
from datetime import date

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# TOOL_MODE=mcp    -> tools come from the MCP server(s) in mcp_servers.json (default)
# TOOL_MODE=direct -> tools are imported straight from tools.py (no server needed)
TOOL_MODE = os.getenv("TOOL_MODE", "mcp").lower()
if TOOL_MODE == "direct":
    from tools import ALL_TOOLS
else:
    from mcp_tools import load_mcp_tools
    ALL_TOOLS = load_mcp_tools()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
DB_PATH = os.getenv("MEMORY_DB", "memory.db")
MAX_CONTEXT_TOKENS = 60_000  # history budget sent to Claude on each call

SYSTEM_PROMPT = """You are a financial market research analyst. Today is {today}.
You research stocks, ETFs, crypto, indices, commodities, currencies, options, futures
and the macro economy.

How to work:
- Plan briefly, then gather data with your tools before answering. Never state prices,
  figures or recent events from memory; fetch them.
- Use structured data tools first (get_quote, get_price_history, get_fundamentals,
  get_etf_holdings, get_crypto_data, get_crypto_market_overview, get_economic_data,
  get_sec_filings), and web_search / get_company_news for news, context and anything else.
- Derivatives: get_options_chain (one expiration, with Greeks), get_options_overview
  (IV term structure, expected moves, put/call ratios across expirations), option_calculator
  (what-if pricing), get_market_snapshot (futures, FX, rates, volatility indices) and
  get_cot_report (futures positioning). Futures and FX prices can also come from get_quote /
  get_price_history with Yahoo symbols like CL=F, GC=F, ES=F, EURUSD=X.
- Options answers: give the expiration and days to expiry, explain Greeks and IV in plain
  words, and mention the main risks (options can expire worthless; selling options can lose
  more than the premium; IV often drops after earnings). Options data is delayed ~15 minutes.
- Call independent tools in parallel when you can (e.g. quotes for several tickers).
- If a tool fails, try another source or say clearly what you couldn't get.
- If a tool result says "limit_reached", don't retry that tool in this answer; follow the
  hint in the message to use another source, and tell the user which data was limited.
- Be economical: fetch what the question needs, and reuse data already in the conversation.

How to answer:
- Lead with the direct answer, then supporting data. Use tables for comparisons.
- Give the date or timestamp of market data, and note that prices move.
- Cite sources: put a short "Sources" list with URLs at the end.
- Separate facts from interpretation. Present bull and bear points where relevant.
- You provide research and information, not personalized investment advice. Don't tell
  the user to buy or sell; you can lay out the factors they'd weigh.
"""

llm = ChatAnthropic(model=MODEL, max_tokens=4096).bind_tools(ALL_TOOLS)


def agent_node(state: MessagesState):
    """Call Claude with the system prompt plus the (trimmed) conversation."""
    history = trim_messages(
        state["messages"],
        max_tokens=MAX_CONTEXT_TOKENS,
        token_counter=count_tokens_approximately,
        strategy="last",
        start_on="human",       # never start mid tool-call exchange
        include_system=False,
    )
    system = SystemMessage(SYSTEM_PROMPT.format(today=date.today().isoformat()))
    response = llm.invoke([system] + history)
    return {"messages": [response]}


def build_graph(db_path: str = DB_PATH):
    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", ToolNode(ALL_TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)  # -> "tools" or END
    builder.add_edge("tools", "agent")

    conn = sqlite3.connect(db_path, check_same_thread=False)
    memory = SqliteSaver(conn)
    return builder.compile(checkpointer=memory), conn


def langfuse_callbacks() -> list:
    """Return a Langfuse tracing callback if keys are configured, else nothing."""
    if os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"):
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    return []


def run_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "callbacks": langfuse_callbacks(),
        "metadata": {"langfuse_session_id": thread_id},  # groups traces per chat
        "recursion_limit": 30,  # max graph steps per question (cost guard)
    }


if __name__ == "__main__":
    # Quick terminal test:  python agent.py
    graph, _ = build_graph()
    cfg = run_config("cli-test")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit", "quit", ""}:
            break
        for update in graph.stream({"messages": [("user", q)]}, cfg, stream_mode="updates"):
            for node, data in update.items():
                msg = data["messages"][-1]
                if node == "agent" and getattr(msg, "tool_calls", None):
                    print("  [tools]", ", ".join(f"{t['name']}({t['args']})" for t in msg.tool_calls))
                elif node == "agent":
                    print("\nAgent:", msg.text)
        if cfg["callbacks"]:
            from langfuse import get_client
            get_client().flush()  # send this question's trace to Langfuse now
