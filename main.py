"""
MAIN FILE - This runs everything

This is where you control the FLOW:
Agent 1 → Agent 2 → Agent 3 → etc.
"""

import json
import config
from agents.agent_first_data_fetcher import run_data_collector_agent
from agents.agent_second_analysis import run_analyst_agent


def create_urban_plan(user_query: str):
    """
    Main orchestration function.
    
    Controls the complete workflow:
    1. Agent 1 collects data
    2. Agent 2 analyzes demand
    3. Agent 3 designs solutions
    4. Agent 4 reviews plan and might iterate between Agents 3 and 4
    5. Agent 5 writes report
    """
    
    print("\n" + "🚀 "*20)
    print("URBAN PLANNER AI - STARTING WORKFLOW")
    print("🚀 "*20 + "\n")
    





    
    # ====== AGENT 1: DATA COLLECTION ======
    print("\n[STEP 1/5] DATA COLLECTION")
    print("-" * 60)

    if config.USE_CACHED_AGENT1:
        # --- TEMPORARY: reuse previously saved facts instead of the ~20-min run.
        # Revert by setting USE_CACHED_AGENT1 = False in config.py ---
        print(f"⏩ Using cached Agent 1 facts from '{config.AGENT1_FACTS_CACHE}' (skipping data collection)")
        with open(config.AGENT1_FACTS_CACHE, "r", encoding="utf-8") as f:
            facts = json.load(f)
        print(f"✓ Loaded {len(facts)} cached facts\n")
    else:
        store = run_data_collector_agent(user_query)
        facts = store.facts
        # Save Agent 1 output
        with open("agent1_output.json", "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print("✓ Agent 1 output saved to agent1_output.json\n")







    # ====== AGENT 2: ANALYSIS ======
    print("\n[STEP 2/5] DEMAND ANALYSIS")
    print("-" * 60)

    analysis = run_analyst_agent(facts)

    # Save Agent 2 output
    with open("agent2_output.json", "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)

    print("✓ Agent 2 output saved to agent2_output.json\n")
    






    # ====== AGENTS 3 () ======
    print("\n[STEP 3/5] DESIGN SOLUTIONS (Agent 3)")
















    print("[STEP 4/5] REVIEW PLAN (Agent 4)")
    print("[STEP 5/5] GENERATE REPORT (Agent 5)")
    print("(To be implemented)\n")
    
    print("🎉 "*20)
    print("WORKFLOW COMPLETE")
    print("🎉 "*20)


if __name__ == "__main__":
    
    # Test with a query
    user_query = "Reduce traffic congestion in Jaipur"
    
    create_urban_plan(user_query)