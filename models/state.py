# this is all about storage from all the different agents, i am using. . . 


"""Data storage classes """

'''------------------AGENT # 1-----------------------'''
class ResearchStore:
    """Stores data collected by Agent 1 (Data Collector)"""
    
    def __init__(self):
        self.wikipedia = {}
        self.serper = {}
        self.tavily={}
        self.documents = []
        self.focused_docs=[] #after chunk and similarity search. . .
        self.facts = []  #facts is a list. . .
    
    def add_wikipedia(self, query, results):
        self.wikipedia[query] = results
        self.documents.extend(results)
    
    def add_serper(self, query, results):
        self.serper[query] = results

    def add_tavily(self, query, results):
        self.tavily[query] = results
    
    def add_documents(self, docs):
        self.documents.extend(docs)

    def add_focus_documents(self,foc_doc):
        self.focused_docs.extend(foc_doc)
    
    
    def add_facts(self, facts):
        self.facts.extend(facts)
    
    # debug helpers. . .

    def summary(self):
        return {
            "wikipedia_queries": len(self.wikipedia),
            "tavily_queries": len(self.tavily),
            "documents": len(self.documents),
            "focused_documents":len(self.focused_docs),
            "facts": len(self.facts)
        }



'''------------------AGENT # 2-----------------------'''
class AnalysisStore:
    """Stores data from Agent 2 (Analyst)"""
    
    def __init__(self):
        self.city_name = ""
        self.mobility_patterns = []
        self.current_demand = {}
        self.future_demand = {}
        self.capacity_gaps = []
        self.bottlenecks = []
        self.priority_corridors = []
        self.deficiencies = ""