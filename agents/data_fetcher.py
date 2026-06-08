from crewai.tools import tool
from groq import Groq

import os
import json
import requests
from bs4 import BeautifulSoup
import fitz
import requests
import tempfile


class ResearchStore:

    def __init__(self):

        self.wikipedia = {}
        self.serper = {}
        self.documents = []
        self.facts = []

    def add_wikipedia(self, query, results):

        self.wikipedia[query] = results

        self.documents.extend(results)

    def add_serper(self, query, results):

        self.serper[query] = results

    def add_documents(self, docs):

        self.documents.extend(docs)

    def add_facts(self,facts):
        self.facts.extend(facts)

    # Debug Helpers

    def summary(self):

        return {
            "wikipedia_queries":
                len(self.wikipedia),

            "serper_queries":
                len(self.serper),

            "documents":
                len(self.documents),

            "facts":
                len(self.facts)
        }


client = Groq(
    api_key=os.environ["GROQ_API_KEY"]
)

def generate_research_queries(user_query: str) -> dict:
    """
    Converts a complex urban transport planning query
    into focused research/search queries.

    Args:
        user_query (str): User's transport planning request

    Returns:
        dict: Structured research queries
    """

    SYSTEM_PROMPT = """
        You are an urban transport research planner and information retrieval expert.

        Your task is to convert a user's urban transport planning request into highly effective search queries for data collection.

        The generated queries will be used with:
        - Wikipedia Search
        - Google Search (Serper)

        Rules:

        1. Generate search-engine friendly queries.
        2. Always include the city name when possible.
        Example:
        - "Lucknow population growth"
        - NOT "population growth"

        3. Prioritize queries that are likely to return:
        - numerical values
        - statistics
        - percentages
        - growth rates
        - ridership figures
        - budgets
        - project costs
        - transport indicators
        - official planning data
        - government reports

        4. Focus on information needed for transport planning:

        - population and demographics
        - population growth and future population projections
        - employment and economic activity
        - land use patterns and major activity centers
        - transport infrastructure
        - public transport performance and ridership
        - travel demand and commuting patterns
        - transport mode share (modal split)
        - vehicle ownership and motorization rates
        - traffic congestion and bottlenecks
        - sustainability and non-motorized transport
        - current and planned transport projects
        - transport policies and mobility initiatives

        5. Avoid vague or ambiguous queries.

        6. Queries should be directly usable in Google or Wikipedia search.

        7. Generate 2 high-quality queries per category.

        Return ONLY valid JSON.

        Output format:

        {
        "demographics": [],
        "economics": [],
        "land_use": [],
        "transport_infrastructure": [],
        "mobility_patterns": [],
        "traffic_congestion": [],
        "environment_sustainability": [],
        "policies_projects": [],
        "financials": []
        }
    """

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_query
            }
        ]
    )

    queries = json.loads(
        response.choices[0].message.content
    )
    

    return queries


question_return = generate_research_queries(
    "Reduce traffic congestion in Bangalore."
)

# print(json.dumps(question_return, indent=3))

all_questions=[]

for key, value in question_return.items():
    for i in value:
        all_questions.append(i)
    
