""" Agent 2:  understanding the problem """

# No external tools — just LLM reasoning over Agent 1's facts.
# NOTE on design: Agent 1 produces ~350 facts (~21k tokens). Sending them all in
# ONE Groq call exceeds the free-tier 8000 tokens-per-minute limit and always
# 429s. So this agent uses a MAP-REDUCE approach:
#   1. dedupe + drop empty docs
#   2. MAP   : batch facts under the TPM ceiling; each call keeps only the
#              transport-relevant facts and discards demographic/tourism noise
#   3. REDUCE: one analysis call over the clean digest -> structured JSON
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
# ~12000 chars ≈ ~3000 tokens of input, leaving room for the prompt + output.
MAP_CHAR_BUDGET = 12000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chat_json(messages: list, max_tokens: int) -> dict:
    """One Groq JSON-mode call with retry on 429 (mirrors Agent 1 throttling)."""
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.LLM_MODEL,
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


def _collect_unique_facts(facts: list) -> list:
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


def _batch_by_chars(items: list, budget: int) -> list:
    """Group items into batches whose joined length stays under `budget`."""
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


def _build_evidence_digest(facts: list) -> list:
    """MAP stage: return a compact, transport-relevant, de-noised fact list."""
    unique = _collect_unique_facts(facts)
    batches = _batch_by_chars(unique, MAP_CHAR_BUDGET)
    print(f"  [analyst] {len(unique)} unique facts -> {len(batches)} map batch(es)")

    digest = []
    for i, batch in enumerate(batches, 1):
        payload = "\n".join(f"- {f}" for f in batch)
        result = _chat_json(
            messages=[
                {"role": "system", "content": _MAP_PROMPT},
                {"role": "user", "content": payload},
            ],
            max_tokens=1500,
        )
        kept = result.get("relevant_facts", [])
        digest.extend(kept)
        print(f"  [analyst] map batch {i}/{len(batches)}: kept {len(kept)}/{len(batch)} facts")
        time.sleep(config.LLM_REQUEST_DELAY)  # smooth TPM usage between calls

    return digest


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

    Flow:
    1. Input: facts collected by Agent 1 (list of per-doc {key_facts, ...})
    2. MAP: dedupe + filter to transport-relevant evidence (TPM-safe batches)
    3. REDUCE: one analysis call -> structured JSON
    '''
    print("  [analyst] building transport-relevant evidence digest...")
    digest = _build_evidence_digest(facts)

    if not digest:
        print("  [analyst] WARNING: no transport-relevant evidence found")
        return {"error": "no transport-relevant evidence extracted from Agent 1 facts"}

    evidence_text = "\n".join(f"- {f}" for f in digest)
    print(f"  [analyst] reducing {len(digest)} evidence facts into structured analysis...")

    analysis = _chat_json(
        messages=[
            {"role": "system", "content": _ANALYSIS_PROMPT},
            {"role": "user", "content": f"EVIDENCE:\n{evidence_text}"},
        ],
        max_tokens=config.LLM_MAX_FACT_TOKENS,
    )
    return analysis
