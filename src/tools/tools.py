
from langchain_tavily import TavilySearch
import os
from langchain.tools import tool
import requests
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


BASE_URL = "https://api.dharun.dev/v1/census"

# Census 2011 state code for Chhattisgarh
CG_STATE_CODE = "22"


@tool
def get_cg_district_population(district: str):
    """
    Get Census 2011 population and demographic information
    for a Chhattisgarh district.

    district: District name, e.g. Raipur, Durg, Bilaspur,
              Mahasamund, Bastar.
    """

    try:
        # Step 1: Search for district
        search_response = requests.get(
            BASE_URL,
            params={
                "q": district,
                "level": "district"
            },
            timeout=10
        )

        search_response.raise_for_status()
        search_body = search_response.json()

        districts = search_body.get("data", [])

        # Keep only Chhattisgarh districts
        cg_districts = [
            item for item in districts
            if item.get("state", {}).get("code") == CG_STATE_CODE
            and item.get("level") == "district"
        ]

        # Step 2: Find exact district name
        matched_district = next(
            (
                item for item in cg_districts
                if item.get("name", "").casefold() == district.casefold()
            ),
            None
        )

        if not matched_district:
            return {
                "error": f"District '{district}' not found in Chhattisgarh.",
                "available_matches": [
                    item.get("name") for item in cg_districts
                ]
            }

        district_code = matched_district["code"]

        # Step 3: Fetch district population
        population_response = requests.get(
            f"{BASE_URL}/district/{district_code}",
            timeout=10
        )

        population_response.raise_for_status()
        population_body = population_response.json()

        return population_body.get("data", {})

    except requests.RequestException as e:
        return {"error": f"Census API error: {str(e)}"}