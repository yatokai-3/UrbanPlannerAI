""" Agent 4: Plan Critic — the senior reviewer that approves or sends back the design.

This is the "second-opinion consultant" of the pipeline (flow.tldr calls it the
VVV IMP agent — "its word and conclusion hold real values"). It is PURE LLM
reasoning: it reads Agent 3's plan (which already carries the deterministic
_computed_evidence) and Agent 2's analysis (the problem to be solved), then:
  - reviews each solution for technical feasibility, financial realism, integration
  - flags weaknesses (over-ambitious timeline, underestimated cost, bad mode fit)
  - suggests improvements (re-route, cost optimisation, alternative mode)
  - returns a STRUCTURED VERDICT that drives the 3<->4 revision loop:
        APPROVED        -> proceed to Agent 5 (report)
        NEEDS_REVISION  -> loop back to Agent 3 with revision_instructions

No tools needed: the numbers it judges are already in Agent 3's output. (A future
"trust-but-verify" version could re-call the ridership/feasibility tools to
independently recompute a figure — see notes in the chat.)
"""

import json
import os
import time
import re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import config
from utils import token_meter


client = Groq(api_key=os.environ["GROQ_API_KEY"])


def chat_json(messages: list, max_tokens: int) -> dict:
    """One Groq JSON-mode call with retry on 429 (same pattern as Agents 1-3)."""
    for attempt in range(config.LLM_MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=config.LLM_JSON_MODEL,  # reliable JSON
                temperature=config.LLM_TEMPERATURE,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                messages=messages,
            )
            token_meter.record(response)
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            msg = str(e)
            if ("rate_limit" in msg or "429" in msg) and attempt < config.LLM_MAX_RETRIES - 1:
                match = re.search(r"try again in ([\d.]+)s", msg)
                wait = float(match.group(1)) + 0.5 if match else config.LLM_REQUEST_DELAY * (attempt + 1)
                print(f"  [critic] rate limited — waiting {wait:.1f}s (attempt {attempt + 1}/{config.LLM_MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError("Groq call failed after all retries")


_CRITIC_PROMPT = """
    You are a SENIOR URBAN-PLANNING REVIEWER with 20+ years of experience.
    Your review carries real weight — be rigorous, specific, and fair.

    You are given:
    - ANALYSIS: the city's transport problem (priority corridors, demand, gaps).
    - PLAN: the designer's proposed solutions, each with computed numbers
      (ridership, peak demand vs mode capacity, capital cost, break-even, timeline)
      under "_computed_evidence". Treat those numbers as given, but JUDGE them.

    REVIEW each proposed solution on:
    1. Technical feasibility — does the mode fit the corridor's demand? Is it
       oversized (wasted capital) or undersized (will be overwhelmed)? Check the
       peak-demand vs peak-capacity numbers.
    2. Financial realism — is the capital cost plausible for the length? Is the
       break-even reasonable, given transit normally needs subsidy?
    3. Integration — do the corridor solutions work together as a network? Gaps?
       Is cycling used sensibly as a feeder?
    4. Whether the plan actually addresses the gaps/corridors in the ANALYSIS.

    IDENTIFY weaknesses explicitly: over-ambitious timelines, underestimated costs,
    weak/unstated assumptions (e.g. the catchment population estimate), poor mode fit.

    SUGGEST improvements: re-routing, cost optimisation, an alternative mode, phasing.

    Then DECIDE:
    - "APPROVED" if the plan is sound enough to proceed to reporting. Minor caveats
      are fine — note them but still approve.
    - "NEEDS_REVISION" if there is a material flaw (wrong mode for the demand,
      unrealistic cost/timeline, fails to address a key corridor). If so, give
      CONCRETE, ACTIONABLE revision_instructions the designer can act on.

    Be decisive — do not request revision for trivial nitpicks.

    Return ONLY valid JSON:
    {
        "overall_verdict": "APPROVED" | "NEEDS_REVISION",
        "overall_assessment": "2-4 sentence summary judgement",
        "solution_reviews": [
            {
                "corridor": "...",
                "recommended_mode": "...",
                "technical_feasibility": "...",
                "financial_realism": "...",
                "integration": "...",
                "verdict": "sound" | "flawed",
                "concerns": ["..."]
            }
        ],
        "key_weaknesses": ["..."],
        "improvement_suggestions": ["..."],
        "revision_instructions": "concrete guidance for the designer; empty string if APPROVED"
    }
"""


def run_critic_agent(plan: dict, analysis: dict) -> dict:
    """
    Agent 4: review Agent 3's plan against Agent 2's analysis.

    Args:
        plan: Agent 3 output (includes _computed_evidence)
        analysis: Agent 2 output (the problem the plan should solve)

    Returns:
        Structured critique with overall_verdict that drives the 3<->4 loop.
    """
    print("  [critic] reviewing the design against the analysis...")
    review = chat_json(
        messages=[
            {"role": "system", "content": _CRITIC_PROMPT},
            {"role": "user", "content": json.dumps({"ANALYSIS": analysis, "PLAN": plan})},
        ],
        max_tokens=3000,
    )
    verdict = review.get("overall_verdict", "NEEDS_REVISION")
    print(f"  [critic] verdict: {verdict}")
    return review


# Run Agent 4 standalone on the cached Agent 2 + Agent 3 outputs:
#   python -m agents.agent_fourth_critic
if __name__ == "__main__":
    import sys
    city = sys.argv[1] if len(sys.argv) > 1 else "Jaipur"

    with open(config.json_path(config.with_city(config.AGENT2_ANALYSIS_CACHE, city)), "r", encoding="utf-8") as f:
        analysis_input = json.load(f)
    with open(config.json_path(config.with_city("agent3_output.json", city)), "r", encoding="utf-8") as f:
        plan_input = json.load(f)

    critique = run_critic_agent(plan_input, analysis_input)

    out = config.json_path(config.with_city("agent4_output.json", city))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(critique, f, indent=2)

    print(f"[OK] Agent 4 output saved to {out}")
