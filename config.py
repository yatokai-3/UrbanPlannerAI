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

# File Paths
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# City Data
DEFAULT_CITIES = ["Lucknow", "Delhi", "Odisha"]

# ---- TEMPORARY: Agent 1 caching for end-to-end testing ----
# Agent 1 takes ~20 min to run. While building the downstream agents (2-5),
# set USE_CACHED_AGENT1 = True to skip Agent 1 in main.py and load previously
# saved facts from AGENT1_FACTS_CACHE instead.
# To REVERT to the full generalized pipeline: set USE_CACHED_AGENT1 = False.
USE_CACHED_AGENT1 = True
AGENT1_FACTS_CACHE = "3. FINAL FACTS based on CHUNKS.json"
# Cached Agent 2 analysis, so Agent 3 can be developed without re-running Agent 2.
AGENT2_ANALYSIS_CACHE = "agent2.1_output.json"