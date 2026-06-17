"""Information extraction using LLM"""

import os
import json
from groq import Groq

from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)


def generate_research_queries(user_query: str) -> dict:
    """Generate search queries from user query"""
    
    SYSTEM_PROMPT = """
        You are a TRANSPORT DATA ANALYST preparing research queries.
        
        Your job: Generate targeted search queries that will return SPECIFIC DATA for demand analysis.
        
        The queries will be used to research a city's transport situation.
        Agent 2 (Analyst) will use the search results to answer these questions:
        - How many trips happen daily? What modes?
        - Where are the bottlenecks?
        - Which corridors need solutions?
        - How many people can't be served by PT?
        
        Therefore, your queries MUST:
        1. Ask for SPECIFIC NUMBERS (not concepts like "congestion")
        2. Target MAJOR CORRIDORS (not generic city questions)
        3. Focus on CURRENT DATA (not future projections)
        4. Seek MODE SHARE data (% using cars, buses, walking)
        5. Ask for INFRASTRUCTURE GAPS (what's missing?)
        
        AVOID generic questions like:
        ❌ "traffic congestion in [city]"
        ❌ "population of [city]"
        ❌ "transportation systems"
        
        DO ask specific questions like:
        ✓ "daily commute trips from residential areas to CBD in [city] 2024"
        ✓ "[corridor name] peak hour traffic volume and mode share"
        ✓ "bus ridership and frequency on major routes in [city]"
        ✓ "commute time from [zone] to [zone] by car vs bus 2024"
        ✓ "railway and metro capacity utilization in [city]"
        
        STRUCTURE YOUR QUERIES BY THESE CATEGORIES:
        
        1. BASELINE MOBILITY METRICS
        (Population, daily trips, growth rate)
        Ask for: total population + recent growth %, daily trip statistics
        Example: "[City] total daily commute trips and growth rate 2020-2024"
        
        2. CURRENT MODE SHARE & RIDERSHIP
        (How many use cars, buses, walking, cycling?)
        Ask for: % using each mode + absolute numbers
        Example: "[City] public transport mode share and ridership data 2024"
        Example: "[City] motorcycle/car usage percentage and daily trips"
        
        3. MAJOR CORRIDORS & OD PAIRS
        (Where do trips happen? City center, residential zones, employment zones)
        Ask for: Peak flows between major zones, commute distances/times
        Example: "commute volume and travel time from [residential zone] to [CBD/IT Park]"
        Example: "[major road/corridor name] daily traffic volume by mode 2024"
        
        4. INFRASTRUCTURE CAPACITY & GAPS
        (What exists? What's missing?)
        Ask for: Bus routes/frequency, metro coverage, road capacity
        Example: "[City] bus network coverage and average frequency per route"
        Example: "metro/train stations coverage areas in [City]"
        Example: "[City] road congestion indices and peak hour delays 2024"
        
        5. TRANSPORT SYSTEM PERFORMANCE
        (How well is existing PT working?)
        Ask for: Bus punctuality, crowding, waiting times, service reliability
        Example: "[City] public bus average occupancy and crowding statistics"
        Example: "bus service reliability and on-time performance in [City] 2024"
        
        6. LAST-MILE & FIRST-MILE CONNECTIVITY
        (How do people reach transit? Can they walk/cycle?)
        Ask for: Walking accessibility to PT, cycling infrastructure, auto-rickshaw usage
        Example: "[City] population within walking distance of bus stops/metro stations"
        Example: "[City] cycle infrastructure coverage and usage statistics"
        Example: "auto-rickshaw density and average trips per day in [City]"
        
        Return ONLY valid JSON. No explanations, just data.
        
        Format:
        {
            "baseline_metrics": [query1, query2],
            "mode_share_ridership": [query1, query2],
            "major_corridors": [query1, query2],
            "infrastructure_gaps": [query1, query2],
            "system_performance": [query1, query2],
            "last_first_mile": [query1, query2]
        }
        
        CRITICAL RULES:
        - Each query should be a single, searchable phrase (10-15 words max)
        - Use specific place names, corridor names, year (2024)
        - Ask for NUMBERS, not descriptions
        - Make queries answerable by web search (Wikipedia, government reports, news)
        - Do NOT ask for predictions or opinions
        - Avoid "impact of" or "effect on" questions - ask for data instead
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

print(json.dumps(generate_research_queries("Reduce traffic congestion in Jaipur"), indent=2))


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


