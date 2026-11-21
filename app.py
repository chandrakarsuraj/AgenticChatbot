import os
import uuid

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage
from langgraph.types import Command

load_dotenv()

DEFAULT_TITLE = "New thread"

st.set_page_config(page_title="हमरबानी हमर गोठ", page_icon="💬", layout="centered")
st.title("हमरबानी हमर गोठ")
st.subheader("Created by Suraj")

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your .env file or environment to send messages.")
    st.stop()

from src.agents.agents import get_threads, graph


def message_text(message) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return "\n".join(part for part in parts if part)
    return str(content)


def message_role(message) -> str:
    return "user" if getattr(message, "type", "") == "human" else "assistant"


def graph_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_trace",
    }


def get_pending_interrupt(thread_id: str):
    """Return a pending graph interrupt for this thread, if one exists."""
    snapshot = graph.get_state(graph_config(thread_id))
    for task in snapshot.tasks:
        if task.interrupts:
            return task.interrupts[0]
    return None


def stream_agent_reply(prompt: str, thread_id: str, status):
    """Stream assistant text and show tool progress, never tool results."""
    config = graph_config(thread_id)
    inputs = {"messages": [HumanMessage(content=prompt)]}
    current_status = "Thinking…"
    tool_labels = {
        "web_search": "Searching the web…",
        "get_cg_district_population": "Looking up population data…",
    }

    def update_status(label: str) -> None:
        nonlocal current_status
        if label != current_status:
            status.update(label=label, state="running", expanded=False)
            current_status = label

    def progress_for_messages(messages) -> str | None:
        for message in messages:
            calls = (
                getattr(message, "tool_calls", None)
                or getattr(message, "tool_call_chunks", None)
                or []
            )
            for call in calls:
                name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
                if name:
                    return tool_labels.get(name, "Working on your request…")
        return None

    for mode, event in graph.stream(
        inputs,
        config,
        stream_mode=["messages", "updates"],
    ):
        if mode == "updates":
            chat_update = event.get("chat_node", {})
            progress = progress_for_messages(chat_update.get("messages", []))
            if progress:
                update_status(progress)
            elif "tools" in event:
                update_status("Preparing your answer…")
            continue

        chunk, metadata = event
        if metadata.get("langgraph_node") != "chat_node":
            continue

        progress = progress_for_messages([chunk])
        if progress:
            update_status(progress)

        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if content:
                update_status("Writing your answer…")
                yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
                        update_status("Writing your answer…")
                        yield text


threads = get_threads()
threads_by_id = {thread["thread_id"]: thread for thread in threads}

if "active_thread_id" not in st.session_state:
    previous_id = st.session_state.get("thread_id")
    if previous_id and previous_id in threads_by_id:
        st.session_state.active_thread_id = previous_id
    elif threads:
        st.session_state.active_thread_id = threads[0]["thread_id"]
    else:
        st.session_state.active_thread_id = str(uuid.uuid4())

active_thread_id = st.session_state.active_thread_id
active_thread = threads_by_id.get(
    active_thread_id,
    {
        "thread_id": active_thread_id,
        "title": DEFAULT_TITLE,
        "messages": [],
        "updated_at": "",
    },
)

with st.sidebar:
    st.subheader("Your chats")
    if st.button("＋ New thread", use_container_width=True):
        st.session_state.active_thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    sidebar_threads = list(threads)
    if active_thread_id not in threads_by_id:
        sidebar_threads.insert(0, active_thread)

    for thread in sidebar_threads:
        thread_id = thread["thread_id"]
        prefix = "● " if thread_id == active_thread_id else ""
        if st.button(
            prefix + thread["title"],
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):
            st.session_state.active_thread_id = thread_id
            st.rerun()

    st.caption("Threads and messages come from the agent's SQLite checkpointer.")

st.caption(active_thread["title"])
for message in active_thread["messages"]:
    # ToolMessage contains raw tool output; keep it in graph state but not in chat UI.
    if getattr(message, "type", "") == "tool":
        continue
    text = message_text(message).strip()
    if not text:
        continue
    with st.chat_message(message_role(message)):
        st.markdown(text)

pending_interrupt = get_pending_interrupt(active_thread_id)
if pending_interrupt:
    with st.chat_message("assistant"):
        st.warning("Web search is waiting for your approval.")
        interrupt_value = getattr(pending_interrupt, "value", pending_interrupt)
        if isinstance(interrupt_value, dict):
            st.json(interrupt_value)
        else:
            st.write(interrupt_value)

        approve_col, reject_col = st.columns(2)
        decision = None
        with approve_col:
            if st.button(
                "Approve web search",
                key=f"approve_search_{active_thread_id}",
                use_container_width=True,
            ):
                decision = True
        with reject_col:
            if st.button(
                "Reject web search",
                key=f"reject_search_{active_thread_id}",
                use_container_width=True,
            ):
                decision = False

        if decision is not None:
            try:
                with st.spinner("Resuming the assistant…"):
                    graph.invoke(
                        Command(resume=decision),
                        graph_config(active_thread_id),
                    )
            except Exception as exc:
                st.error(f"Unable to resume the agent: {exc}")
            else:
                st.rerun()
else:
    if prompt := st.chat_input("छत्तीसगढ़ी मं लिखव (Roman spelling is okay)"):
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status = st.status("Thinking…", expanded=False)
            try:
                st.write_stream(
                    stream_agent_reply(prompt, active_thread_id, status)
                )
            except Exception as exc:
                status.update(label="Unable to finish", state="error", expanded=False)
                st.error(f"Agent error: {exc}")
            else:
                if get_pending_interrupt(active_thread_id):
                    status.update(label="Approval required", state="complete", expanded=False)
                else:
                    status.update(label="Done", state="complete", expanded=False)
                st.rerun()

