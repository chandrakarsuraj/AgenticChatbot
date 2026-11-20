import os
import time
import uuid

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage

load_dotenv()

DEFAULT_TITLE = "New thread"

st.set_page_config(page_title="Agent Chat", page_icon="💬", layout="centered")


def create_thread() -> str:
    thread_id = str(uuid.uuid4())
    st.session_state.threads[thread_id] = {
        "title": DEFAULT_TITLE,
        "messages": [],
        "updated_at": time.time(),
    }
    return thread_id


if "threads" not in st.session_state:
    st.session_state.threads = {}
if not st.session_state.threads:
    st.session_state.active_thread_id = create_thread()

if st.session_state.get("active_thread_id") not in st.session_state.threads:
    st.session_state.active_thread_id = next(iter(st.session_state.threads))

st.title("Agent Chat")

with st.sidebar:
    st.subheader("Your chats")
    if st.button("＋ New thread", use_container_width=True):
        st.session_state.active_thread_id = create_thread()
        st.rerun()

    st.divider()
    sorted_threads = sorted(
        st.session_state.threads.items(),
        key=lambda item: item[1]["updated_at"],
        reverse=True,
    )
    for thread_id, thread in sorted_threads:
        prefix = "● " if thread_id == st.session_state.active_thread_id else ""
        if st.button(
            prefix + thread["title"],
            key=f"thread_{thread_id}",
            use_container_width=True,
        ):
            st.session_state.active_thread_id = thread_id
            st.rerun()

    st.caption("Threads stay available while this Streamlit session is active.")

active_thread_id = st.session_state.active_thread_id
active_thread = st.session_state.threads[active_thread_id]
st.caption(active_thread["title"])

for message in active_thread["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not os.getenv("OPENAI_API_KEY"):
    st.warning("Set OPENAI_API_KEY in your .env file or environment to send messages.")
    st.stop()

# Reuse the existing LangGraph graph, which uses InMemorySaver.
from src.agents.agents import graph


def stream_agent_reply(prompt: str, thread_id: str):
    """Stream a response using this thread's in-memory graph state."""
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


if prompt := st.chat_input("Message your agent"):
    thread = st.session_state.threads[active_thread_id]
    thread["messages"].append({"role": "user", "content": prompt})
    thread["updated_at"] = time.time()

    if thread["title"] == DEFAULT_TITLE:
        title = " ".join(prompt.split()) or DEFAULT_TITLE
        thread["title"] = title[:37] + "..." if len(title) > 40 else title

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
            thread["messages"].append(
                {"role": "assistant", "content": response}
            )
            thread["updated_at"] = time.time()
            st.rerun()
