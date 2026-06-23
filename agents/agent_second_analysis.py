""" Agent 2:  understanding the problem """

# no tool needed for this agent, just LLM analysis.
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
import requests
import os
import config


GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)




def run_analyst_agent(facts: list) -> dict:
    
    '''
    Agent 2: Analyst - Analyzes demand for transportation modes.
    Flow:
    1. Input: facts collected by Agent 1
    2. Analyze mobility patterns, current demand, future demand, capacity gaps, bottlenecks, priority corridors.
    3. Output: structured analysis to guide solution design.    

    '''
    SYSTEM_PROMPT='''
        You are a SENIOR TRANSPORT DEMAND ANALYST with 15+ years experience.
        Your task: ANALYZE the city's transport situation.

        INPUT: Raw facts about city (population, infrastructure, current problems)
        OUTPUT: Detailed understanding of transport gaps and challenges

        DO NOT PROPOSE SOLUTIONS - Just understand the problem.
        DO NOT suggest metro/BRT - That's for next agent.
        DO NOT calculate costs - That's for next agent.

        ANALYZE:

        1. MOBILITY PATTERNS (200 words)
        - How do people currently move?
        - Mode share (% using cars, buses, walking, bikes)
        - Peak hour patterns
        - Key origin-destination pairs
        - Commute times

        2. CURRENT DEMAND (150 words)
        - Total daily trips (Population × 2.5 trips/person/day)
        - Current PT capacity vs demand
        - Unmet demand (gap)
        - Who is underserved (poor, outer areas)

        3. FUTURE DEMAND PROJECTION (150 words)
        - Population growth rate (5 or 10 years)
        - Expected trips in 2030, 2035
        - If motorization increases, demand increases
        - Project with and without PT improvements

        4. CAPACITY GAPS (200 words)
        - How many trips CANNOT be served?
        - Which areas have no PT?
        - Which times are worst?
        - Example: "Current buses serve 1M/day, but 5M need PT"

        5. BOTTLENECKS & CONGESTION (200 words)
        - Specific locations where traffic converges
        - Why are they bottlenecks (geography, limited roads)
        - Severity (commute time impact)
        - Economic cost (time waste, accidents)

        6. PRIORITY CORRIDORS (200 words)
        - Rank top 5 corridors by demand
        - For each: origin → destination, distance, daily demand
        - Why is this corridor important?
        - Current congestion level
        DO NOT suggest solutions yet

        7. PUBLIC TRANSPORT DEFICIENCIES (150 words)
        - Coverage gaps (areas with no buses)
        - Frequency gaps (buses every 30 min instead of 10)
        - Reliability issues
        - Comfort issues
        - Example: "Eastern zone: 200k people, only 3 bus routes, no metro"

        8. DEMAND ELASTICITY (100 words)
        - If PT improves, how many switch from cars?
        - Estimate potential mode shift
        - Example: "If metro added, 30% of car users would switch"

        CRITICAL RULES:
        - USE ONLY PROVIDED FACTS
        - Do NOT invent numbers
        - If data missing, STATE IT
        - Be specific (not "high demand" but "250k daily demand")
        - Each section 150-250 words
        - Return ONLY valid JSON

        Format:
        {
            "mobility_patterns": "...",
            "current_demand": {...},
            "future_demand": {...},
            "capacity_gaps": {...},
            "bottlenecks": [...],
            "priority_corridors": [...],
            "pt_deficiencies": "...",
            "demand_elasticity": "..."
        }
    '''

    response=client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(facts)}
        ]
    )

    analysis=json.loads(response.choices[0].message.content)
    return analysis



