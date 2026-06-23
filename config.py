import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# LLM Settings
# LLM_MODEL = "llama-3.3-70b-versatile"
LLM_MODEL="openai/gpt-oss-120b"
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