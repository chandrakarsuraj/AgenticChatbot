
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_openai import ChatOpenAI
import os
import sqlite3

llm  = ChatOpenAI(model = "gpt-6-luna",  openai_api_key=os.environ.get("OPENAI_API_KEY"),)
class State(MessagesState):
    pass

def chat_node(state:State):

    message  = state["messages"]
    return {"messages" :llm.invoke(message)}


conn = sqlite3.connect(database="chatbot.db", check_same_thread=False)
graph = StateGraph(State)
memory = SqliteSaver(conn)
memory.setup()

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
graph = graph.compile(checkpointer=memory)


def _message_text(message):
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        return " ".join(part for part in parts if part).strip()
    return str(content).strip()


def get_threads():
    """Return checkpointed threads with their latest messages and title."""
    thread_ids = set()
    for checkpoint_tuple in memory.list(None):
        configurable = checkpoint_tuple.config.get("configurable", {})
        thread_id = configurable.get("thread_id")
        if thread_id:
            thread_ids.add(thread_id)

    threads = []
    for thread_id in thread_ids:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = graph.get_state(config)
        messages = (snapshot.values or {}).get("messages", [])
        if not messages:
            continue

        first_user_message = next(
            (message for message in messages if getattr(message, "type", "") == "human"),
            None,
        )
        title = _message_text(first_user_message) if first_user_message else "New thread"
        if len(title) > 40:
            title = title[:37] + "..."

        threads.append(
            {
                "thread_id": thread_id,
                "title": title or "New thread",
                "messages": messages,
                "updated_at": snapshot.created_at or "",
            }
        )

    return sorted(threads, key=lambda thread: thread["updated_at"], reverse=True)

