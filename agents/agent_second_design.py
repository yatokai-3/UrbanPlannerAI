""" Agent 3 -> gives solution design based on analysis from Agent 2 """

import json
import os
import requests
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

def run_solution_design_agent(analysis: dict) -> dict:
    
    '''
    Agent 3: Solution Designer - Designs transport solutions based on analysis.
    Flow:
    1. Input: structured analysis from Agent 2
    2. Design solutions (metro, BRT, bike lanes, etc.) with rationale
    3. Output: detailed solution design with implementation steps and costs.

    '''
    SYSTEM_PROMPT='''
        You are a SENIOR TRANSPORT PLANNER with 15+ years experience.
        Your task: DESIGN a sustainable transport plan for the city.

        INPUT: Structured analysis of transport demand and gaps (from Agent 2)
        OUTPUT: Detailed solution design with rationale, implementation steps, and cost estimates.

        DESIGN SOLUTIONS:

        1. PROPOSED SOLUTIONS (300 words)
        - Based on the analysis, propose specific solutions (e.g., metro line, BRT corridor, bike lanes)
        - For each solution, explain why it's suitable for the city's context and demand patterns

        2. IMPLEMENTATION STEPS (200 words)
        - Outline key steps to implement each solution (planning, funding, construction phases)
        - Identify potential challenges and how to mitigate them

        3. COST ESTIMATES (150 words)
        - Provide rough cost estimates for each proposed solution
        - Consider capital costs, operational costs, and potential funding sources
    '''
    response = client.chat.completions.create(
        model="gpt-4o",
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(analysis)
            }
        ],
        # max_tokens=3000,
        temperature=0.7
    )
    return response.choices[0].message.content
