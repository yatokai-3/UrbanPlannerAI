"""Data storage classes.

Only Agent 1 needs a mutable store: it accumulates data across many steps
(Wikipedia + multiple Tavily searches + documents + focused chunks + facts)
before returning everything at once.

Agents 2-5 do NOT use a store class on purpose — each produces a single clean
output that is passed along as a plain dict and cached as JSON. That dict->JSON
flow is what powers the USE_CACHED_AGENT* workflow, so wrapping it in a class
would add ceremony with no benefit.
"""

'''------------------AGENT # 1-----------------------'''
class ResearchStore:
    """Working buffer for Agent 1 (Data Collector) — built up across the run."""

    def __init__(self):
        self.city = ""           # resolved city name (used for city-wise output filenames)
        self.wikipedia = {}
        self.tavily = {}
        self.documents = []
        self.focused_docs = []   # after chunking + similarity filtering
        self.facts = []          # final extracted facts (Agent 1's output)

    def add_wikipedia(self, query, results):
        self.wikipedia[query] = results
        self.documents.extend(results)

    def add_tavily(self, query, results):
        self.tavily[query] = results

    def add_documents(self, docs):
        self.documents.extend(docs)

    def add_focus_documents(self, foc_doc):
        self.focused_docs.extend(foc_doc)

    def add_facts(self, facts):
        self.facts.extend(facts)

    # debug helper
    def summary(self):
        return {
            "wikipedia_queries": len(self.wikipedia),
            "tavily_queries": len(self.tavily),
            "documents": len(self.documents),
            "focused_documents": len(self.focused_docs),
            "facts": len(self.facts),
        }
