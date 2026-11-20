
from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated
from langgraph.graph import MessagesState
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import os

llm  = ChatOpenAI(model = "gpt-6-luna",  openai_api_key=os.environ.get("OPENAI_API_KEY"),)
class State(MessagesState):
    pass

def chat_node(state:State):

    message  = state["messages"]
    return {"messages" :llm.invoke(message)}

graph = StateGraph(State)
memory = InMemorySaver()

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)
graph = graph.compile(checkpointer=memory)


