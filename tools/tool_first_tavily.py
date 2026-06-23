
# from urllib import response
# from xmlrpc import client
import requests
import json
import os
import tempfile
import fitz
import re
import pdfplumber
from pypdf import PdfReader
from bs4 import BeautifulSoup
from tavily import TavilyClient
from io import BytesIO
import numpy as np
import config


import os
os.environ["HF_HUB_OFFLINE"] = "1"


from sentence_transformers import SentenceTransformer
from llama_index.core.node_parser import SentenceSplitter


from dotenv import load_dotenv
load_dotenv()


embed_model=SentenceTransformer('all-MiniLM-L6-v2')

# print(os.environ.get("tavily_api_key"))

''' THIS FUNCTION JUST FETCHES THE SEARCH RESULTS FROM TAVILY API. 
    IT DOES NOT FETCH THE FULL CONTENT OF THE DOCUMENTS. '''

TAVILY_API_KEY = os.environ["tavily_api_key"]

def tavily_search(query: str, max_result: int = 3) -> list:

    client= TavilyClient(api_key=TAVILY_API_KEY)  
    response = client.search(query=query, 
                            max_results=max_result, 
                            include_answer="advanced",
                            search_depth="advanced",
                            country="india"
                            )
    
    tavily_results = []
    
    for item in response.get("results", []):
        tavily_results.append({
            "title": item.get("title"),
            "link": item.get("url"),
            "score": item.get("score"),
            "snippet": item.get("content"),
            "source": "tavily"
        })
    return tavily_results

# print(json.dumps(tavily_search("Jaipur bus service reliability and on-time performance 2024"), indent=2))


def extract_tables_and_text(pdf_bytes: bytes) -> dict:
    text_parts = []
    table_parts = []
    
    with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
            
            tables = page.extract_tables()
            for table in tables:
                table_parts.append(table)  # list of rows, each a list of cells
    
    return {"text": "\n".join(text_parts), "tables": table_parts}


def fetchfull_tavily_content(tavily_results: list) -> list:    
    """Fetch full content from Tavily search results."""

    final_results = []
    
    for item in tavily_results:
        try:
            tables=None

            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/pdf,*/*",
                "Accept-Language": "en-US,en;q=0.9",
            }
            resp=requests.get(url=item["link"], timeout=10,headers=headers)
            resp.raise_for_status()

            ## PDF case. . .
            if item["link"].lower().endswith(".pdf") or "pdf" in resp.headers.get("Content-Type", "").lower():
                pdf_data=extract_tables_and_text(resp.content)

                text=pdf_data["text"]
                tables=pdf_data["tables"]

                # reader = PdfReader(BytesIO(resp.content))
                # text = "\n".join(page.extract_text() for page in reader.pages)

            ## HTML Case. . . 
            else:
                soup = BeautifulSoup(resp.text, "lxml")
                text = soup.get_text(separator="\n", strip=True)

                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()
                text = soup.get_text(separator="\n", strip=True)
            
            final_results.append({
                "title": item["title"],
                "link": item["link"],
                "score":item.get("score"),
                "full_text": text,
                "tabular_data":tables,
                "source":"tavily"
            })

        except Exception as e:
            print(f"Error fetching content for {item['link']}: {e}")
            final_results.append({
                "title": item["title"],
                "link": item["link"],
                "score":item.get("score"),
                "full_text": f"ERROR: {str(e)}",
                "tabular_data":f"ERROR: {str(e)}",
                "source":"tavily"
            })

    return final_results

# if __name__ == "__main__":
#     query = "Jaipur bus service reliability and on-time performance"
#     step_1 = fetchfull_tavily_content(tavily_search(query))
#     print(json.dumps(step_1, indent=2))








#-------------------HELPERS-------------------



def clean_pdf_text(raw_text:str)->str:
    """Strip repeated headers/footers, TOC lines, and excess whitespace."""
    lines=raw_text.split('\n')

     # Count repeated lines (headers/footers repeat across pages)
    line_counts = {}
    for line in lines:
        stripped = line.strip()
        if len(stripped) > 10:
            line_counts[stripped] = line_counts.get(stripped, 0) + 1
    
    cleaned_lines = [l for l in lines if line_counts.get(l.strip(), 0) < 4]
    text = '\n'.join(cleaned_lines)
    
    # Drop table-of-contents style lines: "Chapter 1 .......... 12"
    text = re.sub(r'.*\.{4,}\s*\d+\s*$', '', text, flags=re.MULTILINE)
    
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()


def chunk_document(text: str, chunk_size: int = 400, overlap: int = 50) -> list:
    """Split cleaned text into overlapping chunks."""
    
    if not text or len(text) < 50:
        return []
    
    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    return splitter.split_text(text)


def filter_chunks_by_similarity(chunks: list, query: str, top_k: int = 20) -> list:
    """Rank chunks by embedding similarity to the query, return top_k."""
    
    if not chunks:
        return []
    
    if len(chunks) <= top_k:
        return chunks  # nothing to filter, return as-is
    
    query_emb = embed_model.encode(query)
    chunk_embs = embed_model.encode(chunks)
    
    # cosine similarity
    sims = chunk_embs @ query_emb / (
        np.linalg.norm(chunk_embs, axis=1) * np.linalg.norm(query_emb) + 1e-8
    )
    
    top_idx = np.argsort(sims)[-top_k:][::-1]
    return [chunks[i] for i in top_idx]






''' now after fetching the full, content from all the pdf and website, we will do the chunking 
and then we will do the embedding and then we will do the vector search and then we will 
do the question answering. '''

def process_documents_for_extraction(tavily_full_results:list, query:str, top_k_chunks:int=5)->list:
    '''
    full pipeline: travily full text result -> fetch full content -> clean -> chunk ->
    filter relevant chunks -> ready for LLM Extraction.
    '''

    fetched_docs = tavily_full_results # for the sake of understanding
    
    processed = []
    
    for doc in fetched_docs:
        if not doc["full_text"] or len(doc["full_text"]) < 100 or doc["full_text"].startswith("ERROR:"):
            continue  # skip empty/failed fetches
        
        cleaned = clean_pdf_text(doc["full_text"])
        chunks = chunk_document(cleaned)
        
        if not chunks:
            continue
        
        relevant_chunks = filter_chunks_by_similarity(chunks, query, top_k=top_k_chunks)
        
        processed.append({
            "title": doc["title"],
            "link": doc["link"],
            "source": doc["source"],
            "score": doc["score"],
            "query": doc.get("query"),  # which search query surfaced this doc
            "content": "\n\n---\n\n".join(relevant_chunks)  # this goes into extract_key_facts
        })
    
    return processed





    
    