
from langchain_tavily import TavilySearch
import os
from langchain.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web with Tavily and return formatted results."""
    if not isinstance(query, str) or not query.strip():
        return "Please provide a non-empty search query."

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Web search is unavailable: set the TAVILY_API_KEY environment variable."

    try:
        tavily_search = TavilySearch(
            tavily_api_key=api_key,
            max_results=5,
            topic="general",
        )
        response = tavily_search.invoke({"query": query.strip()})
        return response.get("results", [])

  
  
    except Exception as e:
        return f"An error occurred during the web search: {e}"