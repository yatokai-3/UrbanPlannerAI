"""Information extraction using LLM"""

import os
import json
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)


def extract_key_facts(documents: list) -> list:
    """
    Extract important facts from documents of SERPER using LLM.
    
    Args:
        documents: List of documents with title, content, source
        
    Returns:
        List with extracted insights and key facts
    """
    
    SYSTEM_PROMPT = """
        You are a senior traffic analyst. Tell me what kind of data are you seeing and study it.
        and tell me your best insights about this.  Straight up insights no unnecessary heading before giving insights.
        
        ALSO,

        Your task is to extract important factual information
        from urban transport and demographic documents.

        Rules:

        Extract transport planning metrics.

        Prioritize following items only when they are present in the content DO NOT made data by YOURSELF:

        - population
        - population growth rate
        - density
        - employment
        - vehicle ownership
        - mode share
        - ridership
        - trip rates
        - congestion indicators
        - major corridors
        - planned projects

    Return ONLY valid JSON:
    {
        "insights": "analysis text",
        "key_facts": ["fact1", "fact2", "fact3"]
    }
    """
    
    extracted_facts = []
    
    for doc in documents:
        title = doc["title"]
        content = doc["content"]
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content}
            ]
        )
        
        facts = json.loads(response.choices[0].message.content)
        
        extracted_facts.append({
            "title": title,
            "source": doc["source"],
            "insights": facts.get("insights", []),
            "key_facts": facts.get("key_facts", [])
        })
    
    return extracted_facts


def generate_research_queries(user_query: str) -> dict:
    """Generate search queries from user query"""
    
    SYSTEM_PROMPT = """
You are an urban transport research planner.

Convert the user's transport request into focused search queries.

Generate 2 high-quality queries per category.

Return ONLY valid JSON:
{
    "demographics": [],
    "economics": [],
    "transport_infrastructure": [],
    "mobility_patterns": [],
    "traffic_congestion": [],
    "policies_projects": []
}
"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    
    queries = json.loads(response.choices[0].message.content)
    return queries


def extract_city_name(user_query: str) -> str:
    """Extract city name from user query"""
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        messages=[
            {"role": "system", "content": "Extract ONLY city name. Nothing else."},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content.strip()