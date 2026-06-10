# this is all about storage from all the different agents, i am using. . . 


"""Data storage classes """

'''------------------AGENT # 1-----------------------'''
class ResearchStore:
    """Stores data collected by Agent 1 (Data Collector)"""
    
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
    
    def add_facts(self, facts):
        self.facts.extend(facts)
    
    # debug helpers. . .

    def summary(self):
        return {
            "wikipedia_queries": len(self.wikipedia),
            "serper_queries": len(self.serper),
            "documents": len(self.documents),
            "facts": len(self.facts)
        }



'''------------------AGENT # 2-----------------------'''
class AnalysisStore:
    """Stores data from Agent 2 (Analyst)"""
    
    def __init__(self):
        self.mobility_patterns = []
        self.current_demand = {}
        self.future_demand = {}
        self.capacity_gaps = []
        self.bottlenecks = []
        self.priority_corridors = []
        self.recommended_focus_areas = []