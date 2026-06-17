"""Serper web search tool"""

import requests
import json
import os
import tempfile
import fitz
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

# print(os.environ.get("serper_api_key"))
SERPER_API_KEY = os.environ["serper_api_key"]


def extract_pdf_text(pdf_url: str) -> str:
    """Extract text from PDF URL"""
    response = requests.get(pdf_url)
    
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(response.content)
        temp_pdf_path = temp_pdf.name
    
    doc = fitz.open(temp_pdf_path)
    full_text = ""
    
    for page in doc:
        full_text += page.get_text()
    
    doc.close()
    os.remove(temp_pdf_path)
    
    return full_text[:5000]


def search_serper(query: str, limit: int = 3) -> list:
    """
    Search using Serper API (Google search).
    
    Args:
        query: Search query
        limit: Number of results
        
    Returns:
        List of search results with link, title, snippet
    """
    
    url = "https://google.serper.dev/search"
    
    payload = {
        "q": query,
        "num": limit,
        "gl": "in"
    }
    
    headers = {
        "X-API-KEY": SERPER_API_KEY,
        "Content-Type": "application/json"
    }
    
    response = requests.post(
        url,
        json=payload,
        headers=headers
    )
    
    data = response.json()
    organic_results = data.get("organic", [])
    
    final_results = []
    
    for result in organic_results:
        final_results.append({
            "title": result.get("title", ""),
            "content": result.get("snippet", ""),
            "link": result.get("link", ""),
            "source": "serper"
        })
    
    return final_results


def fetch_serper_content(serper_results: list, top_k: int = 3) -> list:
    """
    Fetch full webpage content from search results.
    
    Args:
        serper_results: Output from search_serper()
        top_k: Number of results to fetch
        
    Returns:
        List with full page content
    """
    
    top_results = serper_results[:top_k]
    final_results = []
    
    for item in top_results:
        try:
            # PDF case
            if item["link"].lower().endswith(".pdf"):
                text = extract_pdf_text(item["link"])
            
            # HTML case
            else:
                response = requests.get(
                    item["link"],
                    timeout=10,
                    headers={"User-Agent": "Mozilla/5.0"}
                )
                soup = BeautifulSoup(response.text, "lxml")
                
                # Remove unwanted tags
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                
                text = soup.get_text(separator=" ", strip=True)
                text = " ".join(text.split())
            
            final_results.append({
                "title": item["title"],
                "link": item["link"],
                "content": text[:5000],
                "source": "serper"
            })
        
        except Exception as e:
            final_results.append({
                "title": item["title"],
                "link": item["link"],
                "content": f"ERROR: {str(e)}",
                "source": "serper"
            })
    
    return final_results