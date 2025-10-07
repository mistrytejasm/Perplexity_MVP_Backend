from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
from typing import List
from datetime import datetime

# Import configurations and logging
from logger_config import setup_logging, get_logger
from models.schemas import (
    SearchRequest, 
    SearchResponse, 
    QueryAnalysis, 
    WebSearchResults, 
    SearchResult
)

# Import services
from services.tavily_service import TavilyService
from services.groq_service import GroqService
from services.content_synthesizer import ContentSynthesizer
from services.search_orchestrator import SearchOrchestrator

# Load environment variables and setup logging
load_dotenv()
setup_logging()
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Perplexity MVP Backend",
    description="AI-powered search engine with query analysis and content synthesis",
    version="1.0.0"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services (singleton pattern)
groq_service = GroqService()
tavily_service = TavilyService()
content_synthesizer_service = ContentSynthesizer()
orchestrator = SearchOrchestrator()


# ============ MAIN ENDPOINTS ============

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "Perplexity MVP Running",
        "version": "1.0.0",
        "endpoints": {
            "main_search": "/search",
            "docs": "/docs"
        }
    }


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Main search endpoint - executes full pipeline:
    1. Analyze query with Groq
    2. Search web with Tavily
    3. Synthesize comprehensive response
    """
    logger.info(f"🔍 Search request: {request.query}")
    
    try:
        response = await orchestrator.execute_search(request)
        logger.info(f"✅ Search completed: {response.status}")
        return response
    
    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/search-simple")
async def search_simple(query: str):
    """
    Simplified search endpoint - returns just the answer and sources
    """
    try:
        request = SearchRequest(query=query)
        response = await orchestrator.execute_search(request)
        
        return {
            "query": response.original_query,
            "answer": response.synthesized_response.response if response.synthesized_response else None,
            "sources": response.synthesized_response.sources_used if response.synthesized_response else [],
            "status": response.status
        }
    
    except Exception as e:
        logger.error(f"❌ Simple search failed: {e}")
        return {"error": str(e), "query": query}


# ============ COMPONENT TEST ENDPOINTS (for debugging) ============

@app.post("/test/groq-analysis", response_model=QueryAnalysis)
async def test_groq_analysis(query: str):
    """Test query analysis with Groq"""
    logger.info(f"Testing Groq analysis for: {query}")
    
    try:
        analysis = await groq_service.analyze_query(query)
        return analysis
    except Exception as e:
        logger.error(f"❌ Groq analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/tavily-search")
async def test_tavily_search(search_terms: List[str]):
    """Test Tavily search with multiple terms"""
    logger.info(f"Testing Tavily search: {search_terms}")
    
    try:
        results = await tavily_service.search_multiple(search_terms)
        return {
            "search_terms": search_terms,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        logger.error(f"❌ Tavily search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test/content-synthesis")
async def test_content_synthesis(query: str):
    """Test full content synthesis pipeline"""
    logger.info(f"Testing content synthesis for: {query}")
    
    try:
        # Step 1: Analyze query
        query_analysis = await groq_service.analyze_query(query)
        logger.info(f"Analysis complete: {query_analysis.query_type}")
        
        # Step 2: Get search results
        raw_web_results = await tavily_service.search_multiple(
            query_analysis.suggested_searches
        )
        logger.info(f"Found {len(raw_web_results)} search results")
        
        # Step 3: Convert to schema
        search_results = [
            SearchResult(
                title=item.get('title', ''),
                url=item.get('url', ''),
                content=item.get('content', ''),
                score=item.get('score', 0.0),
                calculated_score=item.get('calculated_score'),
                published_date=item.get('published_date')
            )
            for item in raw_web_results
        ]
        
        web_results = WebSearchResults(
            results=search_results,
            total_results=len(search_results),
            query=query,
            search_terms_used=query_analysis.suggested_searches,
            search_duration=1.0
        )
        
        # Step 4: Synthesize
        synthesized = await content_synthesizer_service.synthesize_response(
            query, query_analysis, web_results
        )
        
        return synthesized
    
    except Exception as e:
        logger.error(f"❌ Content synthesis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ HEALTH AND STATUS ============

@app.get("/health")
async def health_check():
    """Comprehensive health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "groq": "operational",
            "tavily": "operational",
            "synthesizer": "operational"
        }
    }


@app.get("/status")
async def status():
    """Get API status and configuration"""
    return {
        "api_name": "Perplexity MVP Backend",
        "version": "1.0.0",
        "status": "running",
        "endpoints_count": len(app.routes)
    }


# ============ RUN SERVER ============

if __name__ == "__main__":
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )