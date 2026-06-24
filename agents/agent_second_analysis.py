""" Agent 2:  understanding the problem """

# No external tools — just LLM reasoning over Agent 1's facts.
# NOTE on design: Agent 1 produces ~350 facts (~21k tokens). Sending them all in
# ONE Groq call exceeds the per-request token ceiling, and Groq's free tier also
# caps tokens-per-day (~100k). So this agent uses TWO-LEVEL filtering before the
# single analysis call:
#   LEVEL 1 (MAP, LLM)   : batch facts; each call semantically keeps only the
#                          transport-relevant ones, discarding demographic/tourism noise
#   LEVEL 2 (SELECT, no-LLM): keyword-score the survivors and cap to a char budget
#                          so the final call fits one request (spends no tokens)
#   REDUCE (LLM)         : one analysis call over the filtered facts -> structured JSON
# The public function signature stays the same: run_analyst_agent(facts) -> dict


import json
import os
import time
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import config


GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)


# Keep each MAP call's evidence well under the per-minute token budget.
# ~8000 chars ≈ ~2000 tokens of input. gpt-oss is a REASONING model — it spends
# completion tokens thinking before emitting JSON, so we leave generous room for
# output (MAP_MAX_TOKENS) and keep input small so input+output stays under 8k TPM.
MAP_CHAR_BUDGET = 8000
MAP_MAX_TOKENS = 4000

# The single REDUCE call must fit input + output under the model's per-request
# token ceiling (llama-3.3-70b free tier = 12000 TPM). So cap the evidence we
# send (~28000 chars ≈ ~7000 tokens) and reserve the rest for the JSON output.
REDUCE_CHAR_BUDGET = 28000
REDUCE_MAX_TOKENS = 3500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chat_json(messages: list, max_tokens: int) -> dict:
    """One Groq JSON-mode call with retry on 429 (mirrors Agent 1 throttling)."""
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
            is_rate_limit = "rate_limit" in msg or "429" in msg
            if is_rate_limit and attempt < config.LLM_MAX_RETRIES - 1:
                match = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(match.group(1)) + 0.5 if match else config.LLM_REQUEST_DELAY * (attempt + 1)
                print(f"  [analyst] rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise  # non-rate-limit error, or out of retries
    raise RuntimeError("Groq call failed after all retries")


def remove_duplicate_facts(facts: list) -> list:
    """Flatten Agent 1's per-doc facts into a de-duplicated list of strings."""
    seen = set()
    unique = []
    for doc in facts:
        for fact in doc.get("key_facts", []):
            text = (fact or "").strip()
            key = text.lower()
            if not text or key in seen:
                continue
            seen.add(key)
            unique.append(text)
    return unique


def facts_batch_maker(items: list, budget: int) -> list:
    """ Group items into batches whose joined length stays under `budget`. """
    batches, current, size = [], [], 0
    for item in items:
        if current and size + len(item) > budget:
            batches.append(current)
            current, size = [], 0
        current.append(item)
        size += len(item)
    if current:
        batches.append(current)
    return batches


# ---------------------------------------------------------------------------
# MAP: filter raw facts down to transport-relevant evidence
# ---------------------------------------------------------------------------

_MAP_PROMPT = """
    You are filtering raw research facts for an urban TRANSPORT demand analysis.

    From the facts below, KEEP only those relevant to transport / mobility planning:
    - population and growth rate, density, employment / economic capacity
    - vehicle ownership, mode share, public-transport ridership
    - trip rates / daily trips, travel times, peak-hour patterns
    - congestion, bottlenecks, accident / safety data
    - road and public-transport network, routes, frequency, capacity
    - major corridors and origin-destination flows
    - planned or under-construction transport projects (metro, BRT, etc.), fares/affordability

    DISCARD irrelevant facts: tourism, religion, language, history, sewerage,
    water supply, fire services, literacy, generic governance.

    Rules:
    - Preserve exact numbers, units, place names and years from the original fact.
    - Do NOT invent, estimate, or merge facts. Copy the relevant ones faithfully.
    - If none are relevant in this batch, return an empty list.

    Return ONLY valid JSON:
    { "relevant_facts": ["fact 1", "fact 2", ...] }
"""


def keep_transport_relevent_chunk(facts: list) -> list:
    """MAP stage: return a compact, transport-relevant, de-noised weird like religion, tourism fact list."""
    unique = remove_duplicate_facts(facts)
    batches = facts_batch_maker(unique, MAP_CHAR_BUDGET)
    print(f"  [analyst] {len(unique)} unique Tranport related facts -> {len(batches)} map batch(es)")

    digest = []
    for i, batch in enumerate(batches, 1):
        payload = "\n".join(f"- {f}" for f in batch)
        try:
            result = chat_json(
                messages=[
                    {"role": "system", "content": _MAP_PROMPT},
                    {"role": "user", "content": payload},
                ],
                max_tokens=MAP_MAX_TOKENS,
            )
            kept = result.get("relevant_facts", [])
            digest.extend(kept)
            print(f"  [analyst] map batch {i}/{len(batches)}: kept {len(kept)}/{len(batch)} crisp tranport facts")
        except Exception as e:
            # Don't let one bad batch abort the whole analysis — skip and continue.
            print(f"  [analyst] map batch {i}/{len(batches)} FAILED, skipping: {e}")
        time.sleep(config.LLM_REQUEST_DELAY)  # smooth TPM usage between calls

    return digest


# ---------------------------------------------------------------------------
# SELECT: deterministic (NO-LLM) fact filtering — saves the daily token budget
# ---------------------------------------------------------------------------

# Transport-relevance keywords. Scoring facts with these is free and instant, so
# we avoid spending the per-day token budget on LLM filtering/condensing.
_TRANSPORT_KEYWORDS = (
    "traffic", "congestion", "vehicle", "car", "bus", "metro", "rail", "train",
    "transit", "transport", "road", "highway", "corridor", "junction", "trip",
    "commut", "ridership", "passenger", "mode share", "peak", "travel time",
    "brt", "cycle", "cycling", "pedestrian", "walk", "parking", "accident",
    "fatalit", "fleet", "route", "frequency", "pcu", "volume", "two-wheeler",
    "motorcycle", "rickshaw", "ipt", "fare", "population", "growth", "density",
    "urbanis", "urbaniz", "employment", "lakh", "per day", "km", "flyover",
    "mobility", "station",
)


def _transport_score(fact: str) -> int:
    """How many transport keywords appear in a fact (0 = likely noise)."""
    low = fact.lower()
    return sum(1 for kw in _TRANSPORT_KEYWORDS if kw in low)


def select_relevant_facts(fact_strings: list, char_budget: int) -> list:
    """
    Level-2 filter: deterministically pick the most transport-relevant facts that
    fit char_budget. Input is a FLAT list of fact strings (the level-1 MAP output).
    NO LLM calls — keyword scoring + char cap. Drops zero-keyword noise first,
    then keeps the highest-scoring facts until the budget is full.
    """
    scored = [(f, _transport_score(f)) for f in fact_strings]
    relevant = [(f, s) for (f, s) in scored if s > 0]          # drop pure noise
    relevant.sort(key=lambda x: x[1], reverse=True)            # most relevant first

    selected, size = [], 0
    for f, _ in relevant:
        line_len = len(f) + 3
        if size + line_len > char_budget:
            continue  # skip overflow; keep scanning for shorter high-score facts
        selected.append(f)
        size += line_len

    print(f"  [analyst] level-2 select: {len(selected)}/{len(fact_strings)} facts kept "
          f"({len(fact_strings) - len(relevant)} dropped as noise, "
          f"{len(relevant) - len(selected)} dropped to fit budget)")
    return selected


# ---------------------------------------------------------------------------
# REDUCE: turn the clean evidence into a structured analysis
# ---------------------------------------------------------------------------

_ANALYSIS_PROMPT = '''
    You are a SENIOR TRANSPORT DEMAND ANALYST with 15+ years of experience.

    TASK: Analyze the city's transport situation from the EVIDENCE provided.
    Your output feeds the next agent (the solution designer), so it must be
    SPECIFIC, QUANTIFIED, and STRUCTURED — not vague prose.

    HARD RULES:
    - DO NOT propose solutions (no metro/BRT recommendations, no cost estimates).
      Your job is to UNDERSTAND and QUANTIFY the problem only.
    - USE ONLY the provided evidence. Do NOT invent numbers.
    - When you state a number, keep its unit, place and year (e.g. "1.2M daily
      trips (2018 CMP)"), not "high demand".
    - If a needed figure is missing from the evidence, do not guess — list it in
      "data_gaps". If you must derive a value (e.g. daily trips ≈ population ×
      trip-rate), state the assumption explicitly in that field.

    Produce these analyses:
    - mobility_patterns: how people move today (mode share, peak patterns, key OD pairs, commute times)
    - current_demand: total daily trips, current PT capacity vs demand, who is underserved
    - future_demand: growth rate and projected trips (5-10 yr), motorization trend
    - capacity_gaps: trips that cannot be served, areas with no PT, worst times
    - bottlenecks: specific congested locations, why, and severity
    - priority_corridors: rank the top corridors by demand
    - pt_deficiencies: coverage / frequency / reliability / comfort gaps
    - demand_elasticity: likely mode shift from private to public transport if PT improves
    - key_metrics: the headline numbers the designer needs, pulled from evidence
    - data_gaps: important figures that were missing from the evidence

    Return ONLY valid JSON in EXACTLY this shape:
    {
        "city": "city name inferred from evidence",
        "key_metrics": {
            "population_current": "value + year",
            "population_growth_rate": "value + period",
            "estimated_daily_trips": "value (+ how derived if estimated)",
            "current_mode_share": "private vehicle / bus / IPT / walk / cycle / rail-metro %, as available",
            "existing_pt_capacity": "value",
            "pt_demand_gap": "value or qualitative gap"
        },
        "mobility_patterns": "...",
        "current_demand": "...",
        "future_demand": "...",
        "capacity_gaps": "...",
        "bottlenecks": [
            {"location": "...", "reason": "...", "severity": "..."}
        ],
        "priority_corridors": [
            {"name": "...", "origin": "...", "destination": "...", "distance_km": "...", "daily_demand": "...", "current_congestion": "..."}
        ],
        "pt_deficiencies": "...",
        "demand_elasticity": "...",
        "data_gaps": ["missing figure 1", "missing figure 2"]
    }
'''


def run_analyst_agent(facts: list) -> dict:
    '''
    Agent 2: Analyst — turns Agent 1's raw facts into a structured understanding
    of the city's transport demand and gaps for the designer (Agent 3).

    Flow (two-level filtering, then analyze):
    1. Input: facts collected by Agent 1 (list of per-doc {key_facts, ...})
    2. LEVEL 1 (MAP, LLM): semantically keep only transport-relevant facts
    3. LEVEL 2 (SELECT, no-LLM): keyword-score + cap to fit one REDUCE call
    4. REDUCE (LLM): one analysis call -> structured JSON
    '''
    # Level 1 — semantic filter via the LLM (dedupes + drops noise by meaning).
    digest = keep_transport_relevent_chunk(facts)
    if not digest:
        print("  [analyst] WARNING: level-1 filter returned no facts")
        return {"error": "no transport-relevant facts extracted from Agent 1 output"}

    # Level 2 — deterministic keyword-score + budget cap so the REDUCE call fits
    # one request (no extra tokens spent).
    selected = select_relevant_facts(digest, REDUCE_CHAR_BUDGET)
    if not selected:
        print("  [analyst] WARNING: level-2 filter returned no facts")
        return {"error": "no transport-relevant facts survived filtering"}

    evidence_text = "\n".join(f"- {f}" for f in selected)
    print(f"  [analyst] reducing {len(selected)} facts into structured analysis (1 LLM call)...")

    analysis = chat_json(
        messages=[
            {"role": "system", "content": _ANALYSIS_PROMPT},
            {"role": "user", "content": f"EVIDENCE:\n{evidence_text}"},
        ],
        max_tokens=REDUCE_MAX_TOKENS,
    )
    return analysis



# Run Agent 2 standalone ONLY when executed directly:
#   python -m agents.agent_second_analysis
# Guarding behind __main__ stops a full run from firing on import (e.g. from
# main.py or test scripts).
if __name__ == "__main__":
    with open(config.AGENT1_FACTS_CACHE, "r", encoding="utf-8") as f:
        facts_crisp = json.load(f)

    analysis_check = run_analyst_agent(facts_crisp)

    with open("agent2_output.json", "w", encoding="utf-8") as f:
        json.dump(analysis_check, f, indent=2)

    print("[OK] Agent 2 output saved to agent2_output.json")


