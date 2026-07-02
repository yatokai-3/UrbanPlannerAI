# UrbanPlannerAI — Project Summary (portable context)

This document is a self-contained brief. Handed to another LLM or collaborator, it
should convey what the project is, how it is built, the engineering decisions, and
the problems that were solved along the way.

## 1. What it is

UrbanPlannerAI is a multi-agent AI system that produces a sustainable urban
transport plan for an Indian city from a single natural-language query
(e.g. "Reduce congestion in Pune"). It runs a five-agent pipeline that collects
real data, quantifies the transport problem, designs mode-specific solutions using
deterministic engineering models, subjects them to an automated critical review,
and emits a professional PDF report.

It is a portfolio project demonstrating applied AI for transport planning. The
design goal throughout was to be logical, defensible, and domain-correct rather
than maximally complex.

## 2. Architecture

Five agents, each a plain Python function, orchestrated in `main.py`:

```
Agent 1  Data Collector   -> researches the city, returns structured facts
Agent 2  Analyst          -> turns facts into a quantified demand analysis
Agent 3  Designer         -> prescribes metro/BRT/cycling per corridor (uses tools)
Agent 4  Critic           -> reviews the plan; APPROVED or NEEDS_REVISION
Agent 5  Reporter         -> renders the approved plan to Markdown + PDF
```

Agents 3 and 4 form a bounded feedback loop: the critic sends the plan back to the
designer with revision instructions until it is APPROVED or `MAX_DESIGN_ROUNDS`
(default 3) is reached.

Key architectural principle: **stateful collection, stateless transforms.**
- Agent 1 mutates a `ResearchStore` (accumulates across many sub-steps).
- Agents 2-5 are pure `dict -> dict` functions whose outputs serialize to JSON.
  This is what makes per-stage caching and city-wise archiving trivial.

Second principle: **the LLM judges, Python computes.** All arithmetic
(ridership, cost, viability, corridor length) lives in deterministic tools. The
LLM is used only for language tasks (query generation, fact extraction, analysis
narrative, mode selection rationale, critique, report prose). This keeps every
number auditable and prevents hallucinated figures.

## 3. Agents in detail

- Agent 1 (`agents/agent_first_data_fetcher.py`): extracts the city and generates
  ~12 targeted research queries; pulls Wikipedia + Tavily results; fetches full
  page/PDF content; cleans, chunks, and embeds documents (all-MiniLM-L6-v2) to keep
  the most relevant chunks; then LLM-extracts structured facts. Output: a list of
  fact objects (~300-350 facts per city). This is the expensive stage (~20 min).

- Agent 2 (`agents/agent_second_analysis.py`): converts raw facts into a structured
  demand analysis (mode share, daily trips, priority corridors with origin/
  destination/demand, bottlenecks, capacity gaps, future demand, data gaps).
  Uses two-level filtering (see problem log) before a single analysis call.

- Agent 3 (`agents/agent_third_design.py`): code-orchestrated in three steps.
  STEP 1 (LLM) estimates per-corridor engineering inputs (length, stops, catchment).
  STEP 1b (deterministic) overrides the LLM's length guess with a real geocoded
  distance and re-derives stop count. STEP 2 (Python) runs ridership + feasibility
  tools for cycling/BRT/metro on every corridor. STEP 3 (LLM) applies the demand
  ladder to recommend one mode per corridor, citing the computed numbers.

- Agent 4 (`agents/agent_fourth_critic.py`): pure-LLM senior reviewer. Reads the
  plan (which carries `_computed_evidence`) and the analysis; checks technical fit,
  financial realism, and integration; returns a structured verdict with
  `revision_instructions` that drive the loop.

- Agent 5 (`agents/agent_fifth_reporting.py`): one LLM call for executive-summary/
  conclusion prose; all tables and figures rendered deterministically from the
  agent JSONs. Emits Markdown always and PDF when `fpdf2` is installed.

## 4. Tools (deterministic, no LLM)

- `tools/tool_third_est_ridership.py` — peak/daily ridership from population
  catchment, stop spacing, coverage factor, mode capture rate, and a car-
  competition factor. Supports bus/metro/BRT/LRT/cycling.
- `tools/tool_third_feasibility.py` — capital cost, O&M, farebox revenue,
  break-even years, peak capacity, implementation timeline, and a viability verdict.
  Cost constants are 2025 Indian benchmarks (metro elevated 250 cr/km, underground
  375, BRT 35, cycling 1.5; validated against a sourced cost PDF).
- `tools/tool_third_geo.py` — resolves real corridor length by geocoding both
  endpoints via OpenStreetMap Nominatim (free, no key) and applying a haversine
  distance x 1.3 detour factor. Falls back to the LLM estimate on failure.
- Agent 1 tools: Wikipedia API, Tavily search + content fetch, LLM extraction.

## 5. The central constraint: Groq free-tier limits

Almost every non-obvious design decision traces back to Groq's free tier:
- Per-request / per-minute token ceilings (~8k for the reasoning model,
  ~12k for the instruct model).
- ~100,000 tokens per day, tracked separately per model.

Two models are used deliberately:
- `LLM_MODEL = openai/gpt-oss-120b` (reasoning) — used by Agent 1 extraction.
- `LLM_JSON_MODEL = llama-3.3-70b-versatile` (instruct) — used everywhere JSON is
  forced, because reasoning models spend the completion budget "thinking" and then
  return empty/truncated JSON.

Measured cost of one fresh end-to-end run is roughly 175k-225k tokens, dominated by
Agent 1 (~100k on the gpt-oss pool). This means a fully live run is not reliably
completable in one day on the free tier, which is why caching and staged runs exist.

## 6. Problem log (what went wrong and how it was fixed)

1. Agent 2 single call too large: sending all ~350 facts (~21k tokens) in one call
   exceeded the per-request limit. Fixed with map-reduce.
2. Reasoning model returned empty JSON: gpt-oss burned the token budget reasoning
   before emitting JSON (`json_validate_failed`, empty generation). Fixed by
   raising `max_tokens` and switching JSON tasks to the llama instruct model.
3. 429 rate limits (per-minute): added retry that honors Groq's "try again in Ns"
   hint, plus inter-call throttling.
4. 413 request-too-large on Agent 2 reduce: the map stage barely filtered, so the
   reduce input was still too big. First tried an LLM "condense" pass...
5. ...which exhausted the per-day budget (too many calls). Root-cause fix: replaced
   LLM filtering with a deterministic keyword-scored filter (zero tokens), giving a
   two-level design: LLM MAP (semantic) then no-LLM SELECT (keyword + budget cap).
6. Agent 2 coverage vs accuracy: after fixing empty fields, the analysis still
   under-used evidence and invented numbers (e.g. a lakh/million unit error;
   "designed capacity" with no source). Fixed by rewriting the analysis prompt to
   treat proposed projects as demand evidence, forbid invented figures, and exhaust
   evidence before writing "not specified".
7. Agent 3 length hallucination: the LLM guessed a 15 km corridor that is actually
   ~4 km. Since cost scales with length, this was the worst error. Fixed by adding
   the Nominatim geocoding tool (validated at 4.2 km).
8. Feasibility over-harshness: farebox-only break-even marked all transit "NOT
   VIABLE". Reframed the designer prompt to treat break-even as financial context,
   not a gate, and to decide primarily on capacity-demand fit at lowest cost.
9. fpdf2 Unicode + layout: core fonts are latin-1 (rupee sign etc. crash) and
   `multi_cell(0,...)` throws when the cursor drifts. Fixed with a sanitizer and
   explicit left-margin resets. Also forced UTF-8 stdout so emoji prints do not
   crash Windows cp1252 consoles.
10. Output hygiene: standardized all filenames to `agentN_output.json`, centralized
    paths (JSON to `outputs/JSON/`, reports to `outputs/`), added city-wise naming
    (`agent1_output_pune.json`) with a tolerant slug so caching survives small
    wording differences, and made `extract_city_name` deterministic (temperature 0).

## 7. Operational model

- Caching: `config.USE_CACHED_AGENT1/2` load cached JSON so downstream stages can be
  developed without re-running the expensive upstream stages.
- Staged runs: run Agent 1 standalone (gpt-oss pool), then `main.py` with Agent 1
  cached (llama pool), so the two daily token pools are hit independently.
- Token meter (`utils/token_meter.py`): records real prompt/completion/total tokens
  per run, split per model, and prints a summary with a rough percentage of the
  100k/day pool.
- City-wise outputs: every artifact is tagged by city, so different cities do not
  overwrite each other. The query in a cached run is used only to select the city.

## 8. Current state and roadmap

Complete and working: all five agents, the 3<->4 loop, deterministic tools,
geocoding, PDF report, caching, token metering, city-wise outputs.

Next step: migrate orchestration to CrewAI (already a dependency). Mapping: agents
to CrewAI Agents, agent functions to Tasks, tools to @tool wrappers. The 3<->4 loop
requires CrewAI Flows (`@router`) since a sequential Crew cannot loop. Caution: keep
Agent 3's deterministic pipeline inside a single custom tool so CrewAI's LLM-driven
tool-calling does not reintroduce non-determinism, and mind that CrewAI is more
token-hungry than the current hand-rolled orchestration.

Known caveats: farebox-only viability is a simplification (real appraisal uses
benefit-cost ratios); corridor catchment population is an estimated assumption;
the free tier effectively limits throughput to roughly one full run per day.
