"""DATA COLLECTOR AGENT - Gathers city information"""

import json
from tools.zero_wikipedia_tool import search_wikipedia
from tools.zero_serper_tool import search_serper, fetch_serper_content
from tools.zero_extraction_tool import (
    extract_key_facts,
    generate_research_queries,
    extract_city_name
)
from models.state import ResearchStore


def run_data_collector_agent(user_query: str) -> ResearchStore:
    """
    Agent 1: Collects data about a city.
    
    Flow:
    1. Parse user query → extract city name + research queries
    2. Search Wikipedia for city profile
    3. Search Serper for specific information
    4. Fetch full webpage content
    5. Extract key facts using LLM
    6. Store everything
    
    Returns:
        ResearchStore with all collected data
    """
    
    print("\n" + "="*60)
    print("AGENT 1: DATA COLLECTOR - STARTING")
    print("="*60 + "\n")
    
    store = ResearchStore()
    
    # Step 1: Generate research queries
    print("📋 Generating research queries...")
    queries = generate_research_queries(user_query)
    
    # Flatten all queries into one list
    all_questions = []
    for category, query_list in queries.items():
        all_questions.extend(query_list)
    
    # print("All the generated question: ", all_questions)
    # print("\n\n")
    print(f"✓ Generated {len(all_questions)} queries\n")
    
    # Step 2: Extract city name
    print("🏙️ Extracting city name...")
    city_name = extract_city_name(user_query)
    print(f"✓ City: {city_name}\n")
    
    # Step 3: Search Wikipedia for city profile
    print(f"🔍 Searching Wikipedia for '{city_name}'...")
    wiki_results = search_wikipedia(city_name + " city", limit=3)
    store.add_wikipedia(city_name, wiki_results)
    print(f"✓ Found {len(wiki_results)} Wikipedia pages\n")
    
    # Step 4: Search Serper for each query
    print(f"🔍 Searching Serper ({len(all_questions)} queries)...")
    for query in all_questions:
        print(f"   - {query}")
        
        serper_results = search_serper(query, limit=3)
        store.add_serper(query, serper_results)
        
        # Fetch full content
        docs = fetch_serper_content(serper_results, top_k=3)
        store.add_documents(docs)
    
    print(f"✓ Completed Serper searches\n")
    
    # Step 5: Extract key facts
    print("🧠 Extracting key facts using LLM...")
    facts = extract_key_facts(store.documents)
    store.add_facts(facts)
    print(f"✓ Extracted facts from {len(store.documents)} documents\n")
    
    print("="*60)
    print("AGENT 1: DATA COLLECTOR - COMPLETE")
    print("="*60)
    print(store.summary())
    
    return store

fir_res=run_data_collector_agent("sustainable plan for banglore")

with open("serper.json","w") as f:
    json.dump(fir_res.serper , f, indent=2)

with open("facts.json","w") as f:
    json.dump(fir_res.facts , f, indent=2)
# After your agent finishes fetching

