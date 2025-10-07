from fastapi import FastAPI
import httpx
import asyncio
from dotenv import load_dotenv
from services.tavily_service import TavilyService
import uvicorn
from logger_config import setup_logging, get_logger
from typing import List
from dotenv import load_dotenv

load_dotenv()

setup_logging()
logger = get_logger(__name__)

app = FastAPI()

@app.get("/")
async def root():
    return {
        "status": "perplexity_MVP Running. :)"
    }

# ----  Tavily service file code test -----------------------
@app.post("/tavily-search")
async def tavily_search(search_term: List[str]):
    search = TavilyService()
    # search_term = ["what is ai", "what is ML"]
    result = await search.search_multiple(search_term)
    print(result)
    return result

# ----------------========================-------------------------------

# ---------- Groq service check ------------
from services.groq_service import GroqService
from models.schemas import QueryAnalysis

@app.post("/groq-service")
async def groq_service_check(query: str) -> QueryAnalysis:
    groq_service = GroqService()
    analysis_query = await groq_service.analyze_query(query)
    print(analysis_query)
    return analysis_query

# -----------------------------------------------------

# ---------------- content synthesizer ----------------
from services.content_synthesizer import ContentSynthesizer
from models.schemas import WebSearchResults, SearchResult

@app.post("/content-synthesizer")
async def content_synthesizer(query: str):
    
    query_analysis = await groq_service_check(query)
    logger.info(f"Analysis Query: {query_analysis}")

    # get raw results from tavily
    raw_web_results = await tavily_search(query_analysis.suggested_searches)
    logger.info(f"Raw web Results: {raw_web_results}")
    logger.info(f"Raw web Results: {len(raw_web_results)} items")

    # convert to websearch_result object
    search_results = []

    for item in raw_web_results:
        search_result = SearchResult(
            title = item.get('title', ''),
            url= item.get('url', ''),
            content = item.get('content', ''),
            score= item.get('score', 0.0)
        )
        search_results.append(search_result)
        logger.info(f"Search_Results: {search_results}")
    

    web_results = WebSearchResults(
        results=search_results,
        total_results=len(search_results),
        query=query,
        search_terms_used= query_analysis.suggested_searches,
        search_duration=1.2
    )
    logger.info(f"Converted to WebSearchResults: {web_results}")

    content = ContentSynthesizer()
    synthesizer = await content.synthesize_response(query, query_analysis, web_results)
    print(f"Synthesizes Content: {synthesizer}")
    return synthesizer


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
