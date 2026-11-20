import os
import uuid

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage

load_dotenv()

DEFAULT_TITLE = "New thread"

st.set_page_config(page_title="suraj", page_icon="💬", layout="centered")
st.title("Suraj's Chatbot")

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


def stream_agent_reply(prompt: str, thread_id: str):
    """Stream a response using the selected checkpointed thread."""
    config = {"configurable": {"thread_id": thread_id}}
    inputs = {"messages": [HumanMessage(content=prompt)]}

    for chunk, metadata in graph.stream(
        inputs,
        config,
        stream_mode="messages",
    ):
        if metadata.get("langgraph_node") != "chat_node":
            continue

        content = getattr(chunk, "content", "")
        if isinstance(content, str):
            if content:
                yield content
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text = block.get("text", "")
                    if text:
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
    with st.chat_message(message_role(message)):
        st.markdown(message_text(message))

if prompt := st.chat_input("Ask Suraj's bot"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(
                stream_agent_reply(prompt, active_thread_id)
            )
        except Exception as exc:
            st.error(f"Agent error: {exc}")
        else:
            st.rerun()
