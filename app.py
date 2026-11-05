"""
Streamlit chat UI.  Run with:  streamlit run app.py
"""

import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from agent import build_graph, run_config
from usage import usage_summary

st.set_page_config(page_title="Market Research Agent", page_icon="📈", layout="wide")


@st.cache_resource
def get_graph():
    return build_graph()  # (graph, sqlite connection), built once per server


graph, conn = get_graph()


def msg_text(msg) -> str:
    """Claude's content can be a string or a list of blocks; pull out the text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text")


def list_threads() -> list[tuple[str, str]]:
    """All saved chats as (thread_id, title), newest first."""
    try:
        rows = conn.execute(
            "SELECT thread_id, MAX(checkpoint_id) FROM checkpoints "
            "GROUP BY thread_id ORDER BY MAX(checkpoint_id) DESC"
        ).fetchall()
    except Exception:
        return []
    threads = []
    for (tid, _) in rows:
        msgs = graph.get_state({"configurable": {"thread_id": tid}}).values.get("messages", [])
        first = next((m for m in msgs if isinstance(m, HumanMessage)), None)
        title = msg_text(first)[:40] if first else "New chat"
        threads.append((tid, title))
    return threads


# ---------------- Sidebar: chat history ----------------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

with st.sidebar:
    st.title("📈 Market Research")
    if st.button("➕ New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()
    st.caption("Previous chats")
    for tid, title in list_threads():
        label = ("▶ " if tid == st.session_state.thread_id else "") + title
        if st.button(label, key=tid, use_container_width=True):
            st.session_state.thread_id = tid
            st.rerun()
    st.divider()
    with st.expander("📊 API usage"):
        fmt = lambda used, cap: f"{used}" if cap is None else f"{used} / {cap}"
        st.table([{
            "Source": r["provider"],
            "Today": fmt(r["day"], r["day_limit"]),
            "This month": fmt(r["month"], r["month_limit"]),
        } for r in usage_summary()])
        st.caption("Limits are set below each free plan. Change them in .env.")
    st.caption("Research and information only, not investment advice.")

config = run_config(st.session_state.thread_id)

# ---------------- Show the current chat ----------------
history = graph.get_state(config).values.get("messages", [])
for m in history:
    if isinstance(m, HumanMessage):
        st.chat_message("user").markdown(msg_text(m))
    elif isinstance(m, AIMessage) and msg_text(m).strip() and not m.tool_calls:
        st.chat_message("assistant").markdown(msg_text(m))

if not history:
    st.markdown("### What do you want to research?")
    st.caption("e.g. *Compare QQQ and VGT* · *How is Solana doing this month?* · "
               "*What did NVIDIA report in its last 10-Q?* · *Where are interest rates heading?*")

# ---------------- Handle a new question ----------------
if question := st.chat_input("Ask about any stock, ETF, crypto, or market..."):
    st.chat_message("user").markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Researching...", expanded=False)
        final = ""
        try:
            for update in graph.stream({"messages": [HumanMessage(question)]}, config,
                                       stream_mode="updates"):
                for node, data in update.items():
                    msg = data["messages"][-1]
                    if node == "agent" and msg.tool_calls:
                        for tc in msg.tool_calls:
                            args = ", ".join(f"{k}={v}" for k, v in tc["args"].items())
                            status.write(f"🔧 `{tc['name']}({args})`")
                    elif node == "agent":
                        final = msg_text(msg)
            status.update(label="Done", state="complete")
        except Exception as e:
            status.update(label="Error", state="error")
            final = f"Something went wrong: `{type(e).__name__}: {e}`"
        st.markdown(final or "_No answer was produced._")
        if config["callbacks"]:
            from langfuse import get_client
            get_client().flush()  # send this question's trace to Langfuse now
