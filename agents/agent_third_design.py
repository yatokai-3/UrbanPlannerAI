""" Agent 3: Solution Designer — prescribes transport solutions for the city.

Design philosophy (the "doctor" of the pipeline):
  - Agent 2 gave the DIAGNOSIS (priority corridors + demand). Agent 3 PRESCRIBES.
  - The MATH is deterministic (Python tools), never the LLM's guess:
        tool_third_est_ridership  -> how many will ride
        tool_third_feasibility    -> cost, timeline, viability verdict
  - The LLM is used ONLY for judgment + narrative, in two small calls:
        STEP 1 (engineer)  : estimate the engineering inputs each corridor needs
                             (length, stops, catchment) — flagged as assumptions
        STEP 3 (synthesis) : read the computed numbers and recommend a mode per
                             corridor (the metro/BRT/cycling demand-ladder)
  - STEP 2 in between is pure Python running the tools on every candidate mode.

This keeps every number auditable and costs only ~2 LLM calls.
"""

import json
import os
import time
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import config

from tools.tool_third_est_ridership import calculate_transit_ridership
from tools.tool_third_feasibility import check_viability
from tools.tool_third_geo import get_corridor_length


client = Groq(api_key=os.environ["GROQ_API_KEY"])

# The three solutions we evaluate, cheapest -> highest-capacity (the demand ladder).
CANDIDATE_MODES = ["cycling", "brt", "metro"]

# Average transit stop/station spacing (km) used to derive stop count from a
# geocoded corridor length (metro ~1.2, BRT ~0.8 — 1.0 is a reasonable mean).
STOP_SPACING_KM = 1.0


# ---------------------------------------------------------------------------
# Shared LLM helper (same retry/throttle pattern as Agents 1 & 2)
# ---------------------------------------------------------------------------

def chat_json(messages: list, max_tokens: int) -> dict:
    """One Groq JSON-mode call with retry on 429. Uses the reliable JSON model."""
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.LLM_JSON_MODEL,  # non-reasoning model: reliable JSON
                temperature=config.LLM_TEMPERATURE,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            msg = str(e)
            if ("rate_limit" in msg or "429" in msg) and attempt < config.LLM_MAX_RETRIES - 1:
                match = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(match.group(1)) + 0.5 if match else config.LLM_REQUEST_DELAY * (attempt + 1)
                print(f"  [designer] rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq call failed after all retries")


# ---------------------------------------------------------------------------
# STEP 1 (LLM) — estimate the engineering inputs each corridor needs
# ---------------------------------------------------------------------------

_ENGINEER_PROMPT = """
    You are a TRANSPORT ENGINEER preparing inputs for ridership and cost models.

    You receive a city transport ANALYSIS (priority corridors + city metrics).
    The corridors usually lack the engineering parameters the models need, so you
    must ESTIMATE them using the city's population/density and sensible Indian
    urban norms, and clearly flag every estimate as an assumption.

    For EACH priority corridor, estimate:
    - route_length_km          : corridor length (from the origin->destination if
                                 known; else a typical urban corridor length)
    - number_of_stops          : stops/stations along it (metro/BRT spacing ~1-1.2 km)
    - population_within_500m    : people living within ~500 m of the corridor
                                 (use city density × a corridor catchment band;
                                  a busy urban corridor often has 100,000-300,000)

    Rules:
    - USE the analysis numbers where given (population, mode share, corridor demand).
    - These are PLANNING ESTIMATES — state how you derived each in "assumptions".
    - Do NOT pick a solution/mode here. That comes later. Just size the corridor.
    - Return numbers as plain integers/floats (no units, no commas).

    Return ONLY valid JSON:
    {
        "city": "city name from analysis",
        "corridors": [
            {
                "name": "corridor name",
                "origin": "...",
                "destination": "...",
                "route_length_km": 12,
                "number_of_stops": 10,
                "population_within_500m": 180000,
                "assumptions": "how length/stops/catchment were derived"
            }
        ]
    }
"""


def estimate_corridor_parameters(analysis: dict) -> dict:
    """STEP 1: ask the LLM to size each corridor (length, stops, catchment)."""
    print("  [designer] STEP 1: estimating corridor engineering parameters...")
    result = chat_json(
        messages=[
            {"role": "system", "content": _ENGINEER_PROMPT},
            {"role": "user", "content": json.dumps(analysis)},
        ],
        max_tokens=2000,
    )
    corridors = result.get("corridors", [])
    print(f"  [designer] sized {len(corridors)} corridor(s)")
    return result


# ---------------------------------------------------------------------------
# STEP 2 (pure Python) — run the deterministic tools for every candidate mode
# ---------------------------------------------------------------------------

def evaluate_corridor(corridor: dict) -> dict:
    """
    For one corridor, run ridership + feasibility for ALL THREE modes.
    No LLM, no guessing — just the tools. Returns a comparison table.
    """
    length = float(corridor.get("route_length_km") or 10)
    stops = max(int(corridor.get("number_of_stops") or 8), 1)
    catchment = int(corridor.get("population_within_500m") or 150000)

    evaluations = {}
    for mode in CANDIDATE_MODES:
        ridership = calculate_transit_ridership(
            route_length_km=length,
            number_of_stops=stops,
            mode_type=mode,
            population_within_500m=catchment,
        )
        feasibility = check_viability(
            mode=mode,
            route_length_km=length,
            daily_ridership=ridership["daily_ridership"],
            peak_hour_ridership=ridership["peak_hour_ridership"],
        )
        evaluations[mode] = {
            "daily_ridership": ridership["daily_ridership"],
            "peak_hour_ridership": ridership["peak_hour_ridership"],
            "peak_capacity_pphpd": feasibility["ridership_metrics"]["mode_peak_capacity_pphpd"],
            "capital_cost_crore": feasibility["financial_summary"]["capital_cost_crore"],
            "annual_opex_crore": feasibility["financial_summary"]["annual_opex_crore"],
            "annual_revenue_crore": feasibility["financial_summary"]["annual_revenue_crore"],
            "break_even_years": feasibility["viability_metrics"]["break_even_years"],
            "implementation_timeline": feasibility["implementation_timeline"],
            "verdict": feasibility["verdict"],
        }
    return {
        "corridor": corridor.get("name"),
        "origin": corridor.get("origin"),
        "destination": corridor.get("destination"),
        "inputs": {"route_length_km": length, "number_of_stops": stops, "population_within_500m": catchment},
        "length_source": corridor.get("length_source", "LLM estimate"),
        "assumptions": corridor.get("assumptions", ""),
        "mode_evaluations": evaluations,
    }


# ---------------------------------------------------------------------------
# STEP 3 (LLM) — recommend a mode per corridor and write the plan
# ---------------------------------------------------------------------------

_DESIGN_PROMPT = """
    You are a SENIOR TRANSPORT PLANNER writing the solution design for a city.

    You are given, for each priority corridor, the COMPUTED numbers for three
    options (cycling, BRT, metro): estimated ridership, peak-hour demand, each
    mode's peak capacity, capital cost, break-even years, timeline, and a
    viability verdict. THESE NUMBERS ARE GROUND TRUTH — do not change them.

    Your job is JUDGMENT, using the demand ladder:
    - metro  : highest-demand corridors (peak ~15,000-70,000 pphpd)
    - BRT    : medium demand (~5,000-20,000 pphpd), far cheaper & faster than metro
    - cycling: short trips / first-last-mile feeder, not a mass-transit line-haul

    HOW TO READ THE VERDICT (important):
    - "break_even_years" and the verdict are FAREBOX cost-recovery indicators only.
      Public transit rarely recovers its capital from fares and is normally
      subsidised — so a long break-even or "NOT VIABLE" tag is NORMAL and must NOT
      by itself rule a mode out. It is financial CONTEXT, not a gate.

    PRIMARY criterion = CAPACITY-DEMAND FIT at lowest capital cost:
    - For each corridor pick the CHEAPEST mode whose peak capacity comfortably
      covers that corridor's peak-hour demand.
    - If peak demand is far below a mode's capacity, that mode is OVERSIZED (wasted
      capital) — step DOWN the ladder (e.g. metro -> BRT).
    - If peak demand exceeds a mode's capacity, step UP (BRT -> metro).
    - Use cycling as the primary recommendation only for genuinely short/feeder
      corridors; otherwise recommend it as a complementary first-last-mile feeder.
    - Always cite the numbers (ridership, peak vs capacity, capital cost, break-even,
      timeline) that justify the choice, and say why the other two were rejected
      (oversized / undersized / too costly for the demand).

    Also give citywide recommendations (e.g. cycling as a feeder network) and list
    the key assumptions/caveats the reviewer (next agent) should check.

    Return ONLY valid JSON:
    {
        "city": "...",
        "corridor_solutions": [
            {
                "corridor": "...",
                "recommended_mode": "metro | brt | cycling",
                "daily_ridership": "...",
                "peak_hour_demand": "...",
                "capital_cost_crore": "...",
                "break_even_years": "...",
                "implementation_timeline": "...",
                "rationale": "why this mode, citing the numbers",
                "alternatives_rejected": "why not the other two modes"
            }
        ],
        "citywide_recommendations": "feeder/cycling network, phasing, integration",
        "assumptions_and_caveats": ["...", "..."]
    }
"""


def run_solution_design_agent(analysis: dict) -> dict:
    """
    Agent 3: turn Agent 2's analysis into a costed transport plan.
    STEP 1 (LLM) size corridors -> STEP 2 (Python) run tools -> STEP 3 (LLM) prescribe.
    """
    # STEP 1 — engineer the inputs
    sized = estimate_corridor_parameters(analysis)
    corridors = sized.get("corridors", [])
    if not corridors:
        return {"error": "no priority corridors to design for"}

    # STEP 1b — override the LLM's guessed length with a real geocoded distance
    # (length drives cost ~linearly, so the LLM's guess is the worst thing to keep).
    # Derive stop count from the real length; fall back to the LLM values if geocoding fails.
    city = sized.get("city", "")
    for c in corridors:
        geo = get_corridor_length(c.get("origin"), c.get("destination"), city)
        if geo and geo.get("length_km"):
            c["route_length_km"] = geo["length_km"]
            c["number_of_stops"] = max(round(geo["length_km"] / STOP_SPACING_KM), 2)
            c["length_source"] = geo["method"]
            print(f"  [designer] geocoded '{c.get('name')}': {geo['length_km']} km "
                  f"(LLM had guessed; now {c['number_of_stops']} stops)")
        else:
            c["length_source"] = "LLM estimate (geocoding unavailable)"

    # STEP 2 — deterministic tool evaluation (no LLM, no tokens)
    print("  [designer] STEP 2: running ridership + feasibility tools (deterministic)...")
    evaluated = [evaluate_corridor(c) for c in corridors]

    # STEP 3 — synthesis / prescription
    print("  [designer] STEP 3: recommending modes and writing the plan...")
    plan = chat_json(
        messages=[
            {"role": "system", "content": _DESIGN_PROMPT},
            {"role": "user", "content": json.dumps({"city": sized.get("city"), "corridor_evaluations": evaluated})},
        ],
        max_tokens=4000,
    )

    # Attach the raw computed evidence so the plan stays auditable downstream.
    plan["_computed_evidence"] = evaluated
    return plan


# Run Agent 3 standalone using the cached Agent 2 analysis (no re-running Agent 2):
#   python -m agents.agent_third_design
if __name__ == "__main__":
    with open(config.AGENT2_ANALYSIS_CACHE, "r", encoding="utf-8") as f:
        analysis_input = json.load(f)

    design = run_solution_design_agent(analysis_input)

    with open("agent3.1_output.json", "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)

    print("[OK] Agent 3 output saved to agent3_output.json")
