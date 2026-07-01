"""
MAIN FILE - This runs everything.

Flow:  Agent 1 (data) -> Agent 2 (analysis) -> [ Agent 3 (design) <-> Agent 4
(critic) loop ] -> Agent 5 (report).

All outputs are saved CITY-WISE so different cities don't overwrite each other:
  Result/JSON/agentN_output_<city>.json   and   Result/transport_plan_<city>.pdf
Upstream agents can be served from cached JSON (config.USE_CACHED_AGENT1/2).
"""

import os
import json
import config
from utils import token_meter
from tools.tool_first_extraction_tool import extract_city_name
from agents.agent_first_data_fetcher import run_data_collector_agent
from agents.agent_second_analysis import run_analyst_agent
from agents.agent_third_design import run_solution_design_agent
from agents.agent_fourth_critic import run_critic_agent
from agents.agent_fifth_reporting import run_report_agent


def _load_cache(path: str, stage: str):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{stage} cache not found: {path}\n"
            f"  Run that stage first (e.g. Agent 1 standalone) or set its USE_CACHED flag to False."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_urban_plan(user_query: str):
    """Run the complete 5-agent workflow for a city query."""

    # Guard: no query -> stop immediately, before any LLM call burns tokens.
    if not user_query or not user_query.strip():
        print("⚠ Please enter a query (e.g. \"Reduce congestion in Jaipur\"). Nothing to do — exiting.")
        return None

    print("\n" + "🚀 " * 20)
    print("URBAN PLANNER AI - STARTING WORKFLOW")
    print("🚀 " * 20 + "\n")

    token_meter.reset()  # start counting Groq tokens for this run

    # Resolve the city up-front so every output file is tagged by city.
    city = extract_city_name(user_query)
    print(f"🏙️  City: {city}  (outputs tagged '_{config.city_slug(city)}')")

    a1_path = config.json_path(config.with_city(config.AGENT1_FACTS_CACHE, city))
    a2_path = config.json_path(config.with_city(config.AGENT2_ANALYSIS_CACHE, city))
    a3_path = config.json_path(config.with_city("agent3_output.json", city))
    a4_path = config.json_path(config.with_city("agent4_output.json", city))

    # ====== AGENT 1: DATA COLLECTION ======
    print("\n[STEP 1/5] DATA COLLECTION")
    print("-" * 60)
    if config.USE_CACHED_AGENT1:
        print(f"⏩ Using cached Agent 1 facts from '{a1_path}'")
        facts = _load_cache(a1_path, "Agent 1")
        print(f"✓ Loaded {len(facts)} cached facts\n")
    else:
        store = run_data_collector_agent(user_query)
        facts = store.facts
        with open(a1_path, "w", encoding="utf-8") as f:
            json.dump(facts, f, indent=2)
        print(f"✓ Agent 1 output saved to {a1_path}\n")

    # ====== AGENT 2: ANALYSIS ======
    print("\n[STEP 2/5] DEMAND ANALYSIS")
    print("-" * 60)
    if config.USE_CACHED_AGENT2:
        print(f"⏩ Using cached Agent 2 analysis from '{a2_path}'")
        analysis = _load_cache(a2_path, "Agent 2")
    else:
        analysis = run_analyst_agent(facts)
        with open(a2_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2)
        print(f"✓ Agent 2 output saved to {a2_path}\n")

    # ====== AGENTS 3 <-> 4: DESIGN / CRITIQUE LOOP ======
    print("\n[STEP 3-4/5] DESIGN <-> CRITIQUE LOOP")
    print("-" * 60)
    plan = run_solution_design_agent(analysis)          # first design
    critique = None
    for round_no in range(1, config.MAX_DESIGN_ROUNDS + 1):
        print(f"\n  --- review round {round_no}/{config.MAX_DESIGN_ROUNDS} ---")
        critique = run_critic_agent(plan, analysis)

        if critique.get("overall_verdict") == "APPROVED":
            print(f"  ✓ Plan APPROVED on round {round_no}")
            break
        if round_no == config.MAX_DESIGN_ROUNDS:
            print("  ⚠ Max rounds reached without approval — proceeding with the last plan")
            break
        print("  ↻ NEEDS_REVISION — sending feedback back to the designer")
        plan = run_solution_design_agent(analysis, critique)   # revise

    with open(a3_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    with open(a4_path, "w", encoding="utf-8") as f:
        json.dump(critique, f, indent=2)
    print(f"✓ Final plan + critique saved ({a3_path}, {a4_path})\n")

    # ====== AGENT 5: REPORT ======
    print("\n[STEP 5/5] GENERATE REPORT (Agent 5)")
    print("-" * 60)
    base = config.report_path(f"transport_plan_{config.city_slug(city)}")
    result = run_report_agent(plan, critique, analysis=analysis, facts=facts, out_basename=base)
    print(f"✓ Report written: {result['markdown_path']}"
          + (f" + {result['pdf_path']}" if result["pdf_path"] else " (Markdown only)"))

    print("\n" + "🎉 " * 20)
    print("WORKFLOW COMPLETE")
    print("🎉 " * 20)

    # How many Groq tokens this whole run actually cost (measured, not estimated).
    token_meter.report("TOTAL TOKEN USAGE FOR THIS RUN")


if __name__ == "__main__":
    import sys
    # Windows consoles default to cp1252 and crash on the emojis we print — force UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    # Query must be provided:  python main.py "Reduce congestion in Pune"
    user_query = sys.argv[1].strip() if len(sys.argv) > 1 else ""
    if not user_query:
        print("⚠ Please enter a query, e.g.:  python main.py \"Reduce congestion in Jaipur\"")
        sys.exit(0)
    create_urban_plan(user_query)
