"""Information extraction using LLM"""
from tools.tool_first_tavily import tavily_search,fetchfull_tavily_content,process_documents_for_extraction

import os
import json
import time
import re
from groq import Groq
import config
from dotenv import load_dotenv
load_dotenv()

# GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY = os.environ["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)


def generate_research_queries(user_query: str) -> dict:
    """Generate search queries from user query"""
    
    SYSTEM_PROMPT = """
        You are a TRANSPORT DATA ANALYST preparing research queries and the current year is 2026.
        
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
        ✓ "daily commute trips from residential areas to CBD in [city] in recent 4-5 years"
        ✓ "[corridor name] peak hour traffic volume and mode share"
        ✓ "bus ridership and frequency on major routes in [city]"
        ✓ "commute time from [zone] to [zone] by car vs bus in recent 4-5 years"
        ✓ "railway and metro capacity utilization in recent 4-5 years"
        
        STRUCTURE YOUR QUERIES BY THESE CATEGORIES:
        
        1. BASELINE MOBILITY METRICS
        (Population, daily trips, growth rate)
        Ask for: total population + recent growth %, daily trip statistics
        Example: "[City] total daily commute trips and growth rate"
        
        2. CURRENT MODE SHARE & RIDERSHIP
        (How many use cars, buses, walking, cycling?)
        Ask for: % using each mode + absolute numbers
        Example: "[City] public transport mode share and ridership data [recent years]"
        Example: "[City] motorcycle/car usage percentage and daily trips"
        
        3. MAJOR CORRIDORS & OD PAIRS
        (Where do trips happen? City center, residential zones, employment zones)
        Ask for: Peak flows between major zones, commute distances/times
        Example: "commute volume and travel time from [residential zone] to [CBD/IT Park]"
        Example: "[major road/corridor name] daily traffic volume by mode [recent years]"
        
        4. INFRASTRUCTURE CAPACITY & GAPS
        (What exists? What's missing?)
        Ask for: Bus routes/frequency, metro coverage, road capacity
        Example: "[City] bus network coverage and average frequency per route"
        Example: "metro/train stations coverage areas in [City]"
        Example: "[City] road congestion indices and peak hour delays [recent years]"
        
        5. TRANSPORT SYSTEM PERFORMANCE
        (How well is existing PT working?)
        Ask for: Bus punctuality, crowding, waiting times, service reliability
        Example: "[City] public bus average occupancy and crowding statistics"
        Example: "bus service reliability and on-time performance in [City] [recent years]"
        
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
        - Use specific place names, corridor names, recent 4-5 years in queries
        - Ask for NUMBERS, not descriptions
        - Make queries answerable by web search (Wikipedia, government reports, news)
        - Do NOT ask for predictions or opinions
        - Avoid "impact of" or "effect on" questions - ask for data instead
    """
    
    response = client.chat.completions.create(
        # model="llama-3.3-70b-versatile",
        # model="openai/gpt-oss-120b",
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_query}
        ]
    )
    
    queries = json.loads(response.choices[0].message.content)
    return queries

# print(json.dumps(generate_research_queries("Reduce traffic congestion in Banglore"), indent=2))


def extract_city_name(user_query: str) -> str:
    """Extract city name from user query"""
    
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        messages=[
            {"role": "system", "content": "Extract ONLY city name. Nothing else."},
            {"role": "user", "content": user_query}
        ]
    )
    
    return response.choices[0].message.content.strip()


def extract_key_facts(documents: list) -> list:
    """
    Extract important facts from documents of TAVILY using LLM.
    
    Args:
        documents: List of documents with title, content, source
        
    Returns:
        List with extracted insights and key facts
    """
    
    SYSTEM_PROMPT = """
        You are a senior traffic analyst studying urban transport and demographic documents.

        Your task: extract DETAILED, CONTEXT-RICH facts — not bare numbers.

        Rules:
        - Each fact must be a full sentence with context (what, where, when, source detail if given).

        BAD:  "Ridership: 19,100 per day"

        GOOD: "Jaipur Metro's two operational lines carry approximately 19,100 passengers per day, a figure the report flags as below projected targets."
        - Only extract what is explicitly present in the content. Do NOT invent or estimate missing values.
        - If a category is genuinely absent from this specific document, omit it from key_facts entirely — 
        do not include a "Not available" placeholder.
        - Prioritize these categories when present: population, population growth rate, density, employment,
        vehicle ownership, mode share, ridership, trip rates, congestion indicators, major corridors,
        planned projects.
        - Extract as many distinct, substantive facts as the content actually supports — there is no fixed
        count, aim for thoroughness over brevity.

        Return ONLY valid JSON:
        {
            "insights": "2-4 sentence analytical summary of what this document reveals and why it matters for transport planning",
            "key_facts": ["detailed fact sentence 1", "detailed fact sentence 2", ...]
        }
    """
    
    extracted_facts = []

    for doc in documents:
        title = doc["title"]
        content = doc["content"]

        # Retry only for rate-limit (429) errors. A truncated/invalid JSON is a
        # different problem (max_tokens too low) and retrying won't fix it.
        for attempt in range(config.LLM_MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=config.LLM_MODEL,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_FACT_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": content}
                    ]
                )

                facts = json.loads(response.choices[0].message.content)

                extracted_facts.append({
                    "title": title,
                    "link": doc.get("link"),
                    "source": doc["source"],
                    "query": doc.get("query"),  # the search query this doc came from
                    "insights": facts.get("insights", ""),
                    "key_facts": facts.get("key_facts", [])
                })
                break  # success — stop retrying this doc

            except Exception as e:
                msg = str(e)
                is_rate_limit = "rate_limit" in msg or "429" in msg
                if is_rate_limit and attempt < config.LLM_MAX_RETRIES - 1:
                    # Groq tells us how long to wait ("try again in 2.75s") — honor it.
                    match = re.search(r"try again in ([\d.]+)s", msg)
                    wait = float(match.group(1)) + 0.5 if match else config.LLM_REQUEST_DELAY * (attempt + 1)
                    print(f"  Rate limited on '{title[:40]}' — waiting {wait:.1f}s (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                print(f"  ERROR extracting facts for '{title}': {e}")
                break  # non-rate-limit error (or out of retries) — give up on this doc

        # Throttle between docs to smooth out token-per-minute usage
        time.sleep(config.LLM_REQUEST_DELAY)

    return extracted_facts




# Standard urban-transport planning metrics. The chunk-similarity filter ranks
# document chunks against these so fact extraction isn't limited to the narrow
# topic of the original search query.
STANDARD_TRANSPORT_METRICS = (
    "population, population growth rate, density, employment, "
    "vehicle ownership, mode share, ridership, trip rates, "
    "congestion indicators, major corridors, planned transport projects"
)


def build_extraction_query(original_query: str) -> str:
    """
    Expand a narrow search-style query into a broad transport-planning
    extraction query, used for chunk similarity filtering (NOT for the Tavily
    search itself — that stays narrow/specific).

    This is a deterministic template: it keeps the original query (which carries
    the city/topic context) and appends the standard metric categories. No LLM
    call — the broadened string only feeds embedding-based chunk ranking, so a
    fixed template is just as effective and saves one Groq call per query.
    """
    return f"{original_query} — {STANDARD_TRANSPORT_METRICS}"




# query="Jaipur bus service reliability and on-time performance"
# step_1=fetchfull_tavily_content(tavily_search(query))
# print(f"Fetched: {len(step_1)} docs, "
#       f"{sum(1 for d in step_1 if d['full_text'].startswith('ERROR:'))} failed")

# print("\n\n")
# print(build_extraction_query(query))
# print("\n\n")

# step_2=process_documents_for_extraction(step_1,enhance_query_for_extraction(query))
# print(f"Processed: {len(step_2)} docs survived cleaning/chunking/filtering")
# for doc in step_2:
#     print(f"\n=== {doc['title'][:50]} ===")
#     print(f"Chunk content length: {len(doc['content'])} chars")
#     print(doc['content'][:500])
#     print("...")


# final_facts=extract_key_facts(step_2)
# print(f"Extracted facts from {len(final_facts)} docs")

# with open("doc_extract_3.json","w") as f:
#     json.dump(step_2,f,indent=2)

# with open("key_facts_3.json","w") as g:
#     json.dump(final_facts,g,indent=2)


# print(final_facts)


