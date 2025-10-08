import httpx
from typing import List, Dict, Any, Optional
from config.settings import settings
import asyncio
import logging

logger = logging.getLogger(__name__)

class TavilyService:
    def __init__(self):
        self.api_key = settings.TAVILY_API_KEY
        self.base_url = "https://api.tavily.com"
        self.timeout = 30

    async def search_multiple(self, search_terms: List[str], max_results_per_search: int = 1) -> List[Dict[str, Any]]:
        """Execute Multiple searches in parallel"""

        logger.info(f"Executing: {len(search_terms)} parallel Searches.")

        # create task for parallel executation.
        tasks = []
        for search_term in search_terms:
            task = self._single_search(search_term, max_results_per_search)
            tasks.append(task)

        logger.info(f"Search Terms for Parallel Executation: {tasks}")

        # Execute all searches in parallel
        search_results = await asyncio.gather(*tasks, return_exceptions=True)
        logger.info(f"Search Results: {search_results}")

        # Process results and handle any exceptions
        all_results = []
        for i, result in enumerate(search_results):
            if isinstance(result, Exception):
                logger.info(f"Search Failed: '{search_terms[i]}': {result}")
                continue

            if result and result.get('results'):
                all_results.extend(result['results'])
        
        # Remove Duplicated And Result
        deduplicated_results = self._deduplicated_results(all_results)
        ranked_results = self._rank_results(deduplicated_results)

        logger.info(f"Found {len(ranked_results)} Unique Results.")
        
        return ranked_results

    async def _single_search(self, query: str, max_results: int) -> Dict[str, Any]:
        """Execute a single search via Tavily API"""

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": "advanced",
            "include_answers": False,
            "include_raw_content": True,
            "max_results": max_results,
            "include_domains": [],
            "exclude_domains": ["youtube.com", "tiktok.com"] # filter out video content
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(f"{self.base_url}/search", json=payload)
                response.raise_for_status()

                result = response.json()
                logger.info(f"Search '{query}' Returned {len(result.get('results', []))} results")

                return result
            
            except httpx.HTTPError as e:
                logger.error(f"Tavily API Error for Query: '{query}': {e}")
                return {'results': []}
            except Exception as e:
                logger.error(f"Unexpected error for query: '{query}' : {e}")
                return {'results': []}
            
    def _deduplicated_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove Duplicate Result based On URL"""
        seen_urls = set()
        unique_urls = []

        for result in results:
            url = result.get('url','')
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_urls.append(result)
        
        return unique_urls
    
    def _rank_results(self, results: List[Dict[str, Any]]) -> List[Dict[str,Any]]:
        """Rank Result by Relevance score and content quality"""

        def calculate_score(result: Dict[str, Any]) -> float:
            score = result.get('score', 0.0)

            # Boost score Based on Content length (more comprehensive = better)
            content_length = len(result.get('content', ''))
            logger.info(f"Content length: {content_length}")

            if content_length > 500:
                score += 1.0
            elif content_length > 200:
                score += 0.5

            # Boost score for reputable domain
            url = result.get('url','').lower()
            reputable_domains = [
                # Encyclopedias & General Knowledge
                'wikipedia.org',
                'britannica.com',
                'stanford.edu',
                'ox.ac.uk',
                'mit.edu',

                # Science & Research
                'nature.com',
                'sciencedirect.com',
                'sciencemag.org',
                'springer.com',
                'jstor.org',

                # Technology & Computing / IT
                'ieee.org',
                'acm.org',
                'arxiv.org',
                'nasa.gov',
                'techcrunch.com',

                # News & Journalism
                'bbc.com',
                'nytimes.com',
                'reuters.com',
                'theguardian.com',
                'washingtonpost.com',

                # Health & Medicine
                'nih.gov',
                'who.int',
                'cdc.gov',
                'mayoclinic.org',
                'clevelandclinic.org',

                # Sports (General)
                'espn.com',
                'skysports.com',
                'sports.yahoo.com',
                'cbssports.com',
                'bleacherreport.com',
                'cricbuzz.com'

                # Cricket (Specialized)
                'espncricinfo.com',
                'icc-cricket.com',
                'cricbuzz.com',
                'wisden.com',
                'skysports.com/cricket',

                # Archives & Libraries
                'archive.org',
                'loc.gov',  # Library of Congress
                'europeana.eu',
                'nationalarchives.gov.uk',
                'worlddigitalibrary.org'
            ]

            if any(domain in url for domain in reputable_domains):
                score += 0.15

            return score
        
        # sort by calculated score (highest first)
        ranked = sorted(results, key=calculate_score, reverse=True)
        logger.info(f"Ranked: {ranked}")

        # Add calculated score to results for debugging
        for result in ranked:
            result['calculated_score'] = calculate_score(result)
        
        return ranked
