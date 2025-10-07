import time
from typing import Dict, Any

from models.schemas import SearchRequest, SearchResponse, SearchResult, WebSearchResults
from services.tavily_service import TavilyService
from services.groq_service import GroqService
from services.content_synthesizer import ContentSynthesizer
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SearchOrchestrator:
    """Main orchestrator that coordinates query analysis and web search"""
    
    def __init__(self):
        self.groq_service = GroqService()
        self.tavily_service = TavilyService()
        self.content_synthesizer = ContentSynthesizer()

    async def execute_search(self, request: SearchRequest) -> SearchResponse:
        """Execute complete search pipeline: Analysis + Web Search + Synthesis"""

        start_time  =  time.time()
        analysis = None
        web_result = None

        try:
            # step 1: Analyze Query (directly with groq)
            logger.info(f"step 1. Analyzing Query: '{request.query}'")
            analysis = await self.groq_service.analyze_query(request.query) # Simplified!

            # step 2: Execute web searches
            logger.info(f"step 2. Executing web search")
            web_result = await self._execute_web_search(analysis, request.query)

            



