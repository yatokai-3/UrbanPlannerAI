# UrbanPlannerAI

A multi-agent system that generates a sustainable urban transport plan for an
Indian city from a single natural-language query. It collects real data,
quantifies the transport problem, designs mode-specific solutions using
deterministic engineering models, subjects them to an automated critical review,
and produces a professional PDF report.

Example:
The city (pune) mentioned in main.py, its only purpose to help save the output files with <_city> name.
The two step run is recommended so that, GROQ free tier API limit of 100,000 TPD (token/day), won't flush all at once.
```
# Stage A: collect data (Agent 1, gpt-oss pool)
python -m agents.agent_first_data_fetcher "plan for Pune"

# Stage B: set USE_CACHED_AGENT1 = True, then run the rest (llama pool)
python main.py "Pune"

```

produces `outputs/transport_plan_jaipur.pdf`.

## Overview

The system runs a five-agent pipeline. Agents 3 and 4 form a bounded revision loop
(the critic returns the plan to the designer until it is approved or a round cap is
reached).


<img width="2728" height="1759" alt="floww" src="https://github.com/user-attachments/assets/93673042-02f1-43f2-b81b-d10078cbcb18" />


| Agent | File | Role | LLM/Tools |
| --- | --- | --- | --- |
| 1 | `agents/agent_first_data_fetcher.py` | Collect city data (Wikipedia + Tavily), extract facts | LLM + search/extraction tools |
| 2 | `agents/agent_second_analysis.py` | Quantify demand, corridors, gaps | LLM only |
| 3 | `agents/agent_third_design.py` | Prescribe metro/BRT/cycling per corridor | LLM + ridership/feasibility/geo tools |
| 4 | `agents/agent_fourth_critic.py` | Review technical/financial/integration soundness | LLM only |
| 5 | `agents/agent_fifth_reporting.py` | Render the report | LLM (prose) + fpdf2 |

## Design principles

1. Stateful collection, stateless transforms. Agent 1 accumulates data in a
   `ResearchStore`; Agents 2 to 5 are pure `dict -> dict` functions whose outputs
   serialize to JSON, which makes per-stage caching and city-wise archiving simple.
2. The LLM judges, Python computes. All arithmetic (ridership, cost, viability,
   corridor length) lives in deterministic tools; the LLM handles only language
   tasks. Every number in the final report is auditable, not hallucinated.
3. Demand-ladder solution matching. The designer recommends the cheapest mode whose
   peak capacity covers a corridor's peak demand: metro for very high demand,
   BRT for medium, cycling for short/feeder trips.

## Tools

- `tools/tool_third_est_ridership.py` — daily and peak-hour ridership from
  catchment population, stop spacing, mode capture rate, and a car-competition
  factor (bus/metro/BRT/LRT/cycling).
- `tools/tool_third_feasibility.py` — capital cost, O&M, farebox revenue,
  break-even years, peak capacity, implementation timeline, and viability verdict,
  using 2025 Indian cost benchmarks.
- `tools/tool_third_geo.py` — real corridor length via OpenStreetMap Nominatim
  geocoding (free, no key) plus a haversine distance with a 1.3 detour factor.
- `tools/tool_first_wikipedia.py`, `tools/tool_first_tavily.py`,
  `tools/tool_first_extraction_tool.py` — Agent 1 data collection and extraction.

## Project structure

```
main.py                     orchestration + 3<->4 loop
config.py                   models, token limits, cache flags, path helpers
models/state.py             ResearchStore (Agent 1 working buffer)
utils/token_meter.py        per-run, per-model token accounting
agents/                     the five agents
tools/                      deterministic tools + data-collection tools
outputs/JSON/               per-agent JSON, tagged by city
outputs/                    final reports (transport_plan_<city>.pdf / .md)
```

## Requirements and setup

- Python 3.12
- A Groq API key and a Tavily API key.

```
python -m venv beni
beni\Scripts\activate            # Windows
pip install -r requirements.txt
```

Create a `.env` file in the project root (do not commit it):

```
GROQ_API_KEY=your_groq_key
tavily_api_key=your_tavily_key
```

Note: the Tavily variable name is lowercase (`tavily_api_key`), matching the code.

Optional: `pip install fpdf2` to enable PDF output (Markdown is always produced).

## Configuration (`config.py`)

- `LLM_MODEL` — reasoning model (gpt-oss-120b), used for Agent 1 extraction.
- `LLM_JSON_MODEL` — instruct model (llama-3.3-70b), used wherever JSON is forced.
- `USE_CACHED_AGENT1`, `USE_CACHED_AGENT2` — load a cached upstream stage instead of
  re-running it.
- `MAX_DESIGN_ROUNDS` — cap on designer/critic revision rounds (default 3).
- `LLM_REQUEST_DELAY`, `LLM_MAX_RETRIES`, `LLM_MAX_FACT_TOKENS` — throttling and
  retry behavior for the free-tier rate limits.

## Usage

Every output is tagged by city, so different cities never overwrite each other.

Full run for a new city (data collection runs live):

```
python main.py "Reduce congestion in Pune"
```

Staged run (recommended on the free tier, splits the two model token pools):

```
# Stage A: collect data (Agent 1, gpt-oss pool)
python -m agents.agent_first_data_fetcher "plan for Pune"

# Stage B: set USE_CACHED_AGENT1 = True, then run the rest (llama pool)
python main.py "Pune"
```

In a cached run the query is only used to select the city, so `"Pune"` is enough.

Each agent can also be run standalone with an optional city argument, e.g.
`python -m agents.agent_third_design Jaipur`.

## Outputs

```
outputs/JSON/agent1_output_<city>.json     collected facts
outputs/JSON/agent2_output_<city>.json     demand analysis
outputs/JSON/agent3_output_<city>.json     plan (+ computed evidence)
outputs/JSON/agent4_output_<city>.json     critic verdict
outputs/transport_plan_<city>.pdf / .md    final report
```

## Cost and rate limits

The system is built around Groq free-tier limits (per-request/per-minute token
ceilings and roughly 100,000 tokens per day, tracked separately per model). Two
models are used deliberately: the reasoning model for extraction and the instruct
model for JSON tasks (reasoning models tend to exhaust the completion budget and
return invalid JSON).

Indicative measured costs:
- Agents 2 to 5 on a cached city: about 65,000 tokens (llama pool).
- A full fresh run including live data collection: roughly 175,000 to 225,000
  tokens, dominated by Agent 1 on the gpt-oss pool.

`utils/token_meter.py` prints the real per-model token usage at the end of a run.
Because a full fresh run can approach the daily limit, use the staged workflow and
the caching flags during development.

## Limitations

- Farebox-only break-even is a simplification; real appraisal uses benefit-cost
  ratios that include time savings and externalities.
- Corridor catchment population is an estimated planning assumption.
- Free-tier token limits effectively cap throughput at roughly one full run per day.

## Roadmap

- Migrate orchestration to CrewAI (Agents, Tasks, and @tool wrappers), using CrewAI
  Flows with a router to model the designer/critic loop.
- Streamlit front end (requires a paid tier or pre-cached cities to be practical).
