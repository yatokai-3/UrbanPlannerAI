"""DATA COLLECTOR AGENT - Gathers city information"""

import json
from tools.tool_first_wikipedia import search_wikipedia
# from tools.tool_first_serper import search_serper, fetchfull_serper_content
from tools.tool_first_extraction_tool import *
from models.state import ResearchStore


def run_data_collector_agent(user_query: str) -> ResearchStore:
    """
    Agent 1: Collects data about a city.
    
    Flow:
    1. Parse user query → extract city name + research queries
    2. Search Wikipedia for city profile
    3. Search Tavily for specific information
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
    print(f"✓ Generated {len(all_questions)} queries\n")
    



    # Step 2: Extract city name
    print("🏙️ Extracting city name...")
    city_name = extract_city_name(user_query)
    store.city = city_name   # expose for city-wise output filenames
    print(f"✓ City: {city_name}\n")
    



    # Step 3: Search Wikipedia for city profile
    print(f"🔍 Searching Wikipedia for '{city_name}'...")
    wiki_results = search_wikipedia(city_name + " city", limit=3)
    store.add_wikipedia(city_name, wiki_results)
    print(f"✓ Found {len(wiki_results)} Wikipedia pages\n")

    # Run Wikipedia pages through the same clean/chunk/filter pipeline as Tavily
    # so their content also lands in focused_docs and gets fact-extracted.
    # search_wikipedia returns {title, content, source}; reshape to the doc
    # shape process_documents_for_extraction expects (full_text, link, score).
    wiki_docs = [
        {
            "title": r["title"],
            "link": f"https://en.wikipedia.org/wiki/{r['title'].replace(' ', '_')}",
            "score": None,
            "query": f"{city_name} city",  # the Wikipedia search term used
            "full_text": r.get("content", ""),
            "source": "wikipedia",
        }
        for r in wiki_results
    ]
    wiki_focus = process_documents_for_extraction(wiki_docs, build_extraction_query(city_name))
    store.add_focus_documents(wiki_focus)
    print(f"✓ Added {len(wiki_focus)} Wikipedia docs to focused set\n")



    # Step 4: Search Serper for each query, basically we are not sending a single query in wikipedia. . . 
    # print(f"🔍 Searching Serper ({len(all_questions)} queries)...")
    # for query in all_questions:
    #     print(f"   - {query}")
        
    #     serper_results = search_serper(query, limit=3)
    #     store.add_serper(query, serper_results)
        
    #     # Fetch full content
    #     docs = fetchfull_serper_content(serper_results, top_k=3)
    #     store.add_documents(docs)
    # print(f"✓ Completed Serper searches\n")


    #Step4: Tavily search replacing the serper search
    print(f"🔍 Searching Tavily ({len(all_questions)} queries)...")
    for query in all_questions:
        print(f"   - {query}")
        
        step_1 = tavily_search(query)
        store.add_tavily(query, step_1)

        # Fetch full content
        step_2 = fetchfull_tavily_content(step_1)
        # Tag each fetched doc with the query that surfaced it; this flows
        # through process_documents_for_extraction into the focused docs/facts.
        for d in step_2:
            d["query"] = query
        store.add_documents(step_2)

        #update the query . . .
        enhance_query=build_extraction_query(query)
        foc_doc=process_documents_for_extraction(step_2,enhance_query)
        store.add_focus_documents(foc_doc)


    print(f"✓ Completed Tavily searches\n")


    # Step 5: Extract key facts
    print("🧠 Extracting key facts using LLM...")
    facts = extract_key_facts(store.focused_docs)
    store.add_facts(facts)
    print(f"✓ Extracted facts from {len(store.focused_docs)} documents\n")
    
    print("="*60)
    print("AGENT 1: DATA COLLECTOR - COMPLETE")
    print("="*60)
    print(store.summary())
    
    return store

# Run Agent 1 standalone (and regenerate the cached JSONs) ONLY when this file
# is executed directly: `python -m agents.agent_first_data_fetcher`.
# Guarding behind __main__ means importing this module (e.g. from main.py) does
# NOT trigger a full ~20-min data-collection run.
if __name__ == "__main__":
    import sys
    import config

    # Allow a custom query:  python -m agents.agent_first_data_fetcher "plan for Pune"
    user_query = sys.argv[1] if len(sys.argv) > 1 else "sustainable plan for Jaipur"

    fir_res = run_data_collector_agent(user_query)
    city = fir_res.city          # already resolved inside Agent 1 — no extra LLM call

    # City-wise filenames so different cities don't overwrite each other.
    # Two debug dumps (intermediate stages) + the real output (facts), all in Result/JSON/.
    with open(config.json_path(config.with_city("agent1_documents.json", city)), "w", encoding="utf-8") as f:
        json.dump(fir_res.documents, f, indent=2)

    with open(config.json_path(config.with_city("agent1_focused_docs.json", city)), "w", encoding="utf-8") as f:
        json.dump(fir_res.focused_docs, f, indent=2)

    out = config.json_path(config.with_city("agent1_output.json", city))   # facts = Agent 1's real output
    with open(out, "w", encoding="utf-8") as f:
        json.dump(fir_res.facts, f, indent=2)

    print(f"[OK] Agent 1 saved city-wise outputs for '{city}' -> {out} (+ documents, focused_docs)")


