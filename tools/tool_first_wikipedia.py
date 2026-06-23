"""Wikipedia search tool"""

import requests
import json
from dotenv import load_dotenv
load_dotenv()


WIKIPEDIA_HEADERS = {
    "User-Agent": "UrbanTransportAgent/1.0 (your-email@gmail.com)"
}

WIKIPEDIA_BASE_URL = "https://en.wikipedia.org/w/api.php"


def search_wikipedia(query: str, limit: int = 3) -> list:
    """
    Searches Wikipedia and returns relevant pages with extracts.
    
    Args:
        query: Search query (e.g., "Lucknow city")
        limit: Number of results
        
    Returns:
        List of dicts with title, content, source
    """
    
    # Step 1: Search Wikipedia
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": limit
    }
    
    response = requests.get(
        WIKIPEDIA_BASE_URL,
        headers=WIKIPEDIA_HEADERS,
        params=search_params
    )
    
    data = response.json()
    search_results = data["query"]["search"]
    
    final_results = []
    
    # Step 2: Get full extracts
    for result in search_results:
        title = result["title"]
        
        extract_params = {
            "action": "query",
            "prop": "extracts",
            "titles": title,
            "explaintext": 1,
            "format": "json"
        }
        
        extract_response = requests.get(
            WIKIPEDIA_BASE_URL,
            headers=WIKIPEDIA_HEADERS,
            params=extract_params
        )
        
        extract_data = extract_response.json()
        pages = extract_data["query"]["pages"]
        page_data = list(pages.values())[0]
        extract = page_data.get("extract", "")
        
        final_results.append({
            "title": title,
            "content": extract[:10000],
            "source": "wikipedia"
        })
    
    return final_results
# print(json.dumps(search_wikipedia("Lucknow city"),indent=2))