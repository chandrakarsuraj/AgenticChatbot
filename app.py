import os
import uuid

from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import HumanMessage

load_dotenv()

st.set_page_config(page_title="Agent Chat", page_icon="💬", layout="centered")
st.title("Agent Chat")
st.caption("Chat with your LangGraph agent. Replies stream as they are generated.")

if not os.getenv("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY is not set. Add it to a .env file or your environment, then restart the app.")
    st.stop()

# Import after loading environment variables and checking the API key. The agent
# module builds the compiled LangGraph graph used by this Streamlit frontend.
from src.agents.agents import graph

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Conversation")
    if st.button("New chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    st.caption("Conversation memory lasts while the app process is running.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


def stream_agent_reply(prompt: str):
    """Stream this turn from the existing graph and its session thread."""
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
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


if prompt := st.chat_input("Send a message"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            response = st.write_stream(stream_agent_reply(prompt))
        except Exception as exc:
            st.error(f"Agent error: {exc}")
        else:
            st.session_state.messages.append(
                {"role": "assistant", "content": response}
            )
