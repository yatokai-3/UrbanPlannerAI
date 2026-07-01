import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# LLM Settings
# LLM_MODEL = "llama-3.3-70b-versatile"
'''THIS LLM is a reasoning model hence was causing issue with agent 2 fact'''
LLM_MODEL="openai/gpt-oss-120b"
# Non-reasoning model for JSON-extraction/filtering tasks. Reasoning models burn
# the completion-token budget "thinking" and return empty/truncated JSON, so use
# a plain instruct model wherever we force response_format=json_object.
LLM_JSON_MODEL = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
# Seconds to wait between successive Groq calls (rate-limit throttle)
LLM_REQUEST_DELAY = 2
# Max completion tokens for fact extraction. Too low truncates the JSON mid-way
# and Groq's JSON validator rejects it. Keep input + this under the TPM limit.
LLM_MAX_FACT_TOKENS = 4000
# How many times to retry a single doc after a 429 rate-limit error
LLM_MAX_RETRIES = 5

# ---- Output locations ----
# JSON artifacts go in outputs/JSON/, reports (md/pdf) go in outputs/.
OUTPUT_DIR = "outputs"
JSON_DIR = os.path.join(OUTPUT_DIR, "JSON")
os.makedirs(JSON_DIR, exist_ok=True)


def city_slug(city: str) -> str:
    """
    Normalise a city name into a stable filename slug, tolerant of small wording
    differences so caching is robust:
      'Jaipur' / 'Jaipur City' / 'Jaipur, Rajasthan' -> 'jaipur'
    """
    name = (city or "city").strip().lower()
    name = name.split(",")[0]            # drop ", Rajasthan" / ", India"
    name = name.replace(" city", "")     # drop a trailing "city" qualifier
    s = "".join(c if c.isalnum() else "_" for c in name)
    return s.strip("_") or "city"


def with_city(filename: str, city: str) -> str:
    """'agent1_output.json' + 'Jaipur' -> 'agent1_output_jaipur.json'."""
    base, ext = os.path.splitext(filename)
    return f"{base}_{city_slug(city)}{ext}"


def json_path(filename: str) -> str:
    """Full path for a JSON artifact inside outputs/JSON/."""
    return os.path.join(JSON_DIR, filename)


def report_path(filename: str) -> str:
    """Full path for a report (md/pdf) inside outputs/."""
    return os.path.join(OUTPUT_DIR, filename)

# City Data
DEFAULT_CITIES = ["Lucknow", "Delhi", "Odisha"]

# ---- TEMPORARY: Agent 1 caching for end-to-end testing ----
# Agent 1 takes ~20 min to run. While building the downstream agents (2-5),
# set USE_CACHED_AGENT1 = True to skip Agent 1 in main.py and load previously
# saved facts from AGENT1_FACTS_CACHE instead.
# To REVERT to the full generalized pipeline: set USE_CACHED_AGENT1 = False.
USE_CACHED_AGENT1 = True
AGENT1_FACTS_CACHE = "agent1_output.json"
# Cached Agent 2 analysis, so Agents 3-5 can be developed without re-running Agent 2.
USE_CACHED_AGENT2 = False
AGENT2_ANALYSIS_CACHE = "agent2_output.json"

# Max designer<->critic revision rounds before proceeding with the last plan.
# Each round costs ~2 LLM calls — keep small on the free tier.
MAX_DESIGN_ROUNDS = 3