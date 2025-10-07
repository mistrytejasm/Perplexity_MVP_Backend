import json
from typing import List, Dict, Any, Optional
from groq import AsyncGroq
from models.schemas import WebSearchResults, SearchResult, QueryAnalysis, SynthesizedResponse
from config.settings import settings
import logging
import re

logger = logging.getLogger(__name__)

class ContentSynthesizer:
    """Synthesizer search result into comprehansive, cited responses"""

    def __init__(self):
        self.client = AsyncGroq(api_key=settings.GROQ_API_KEY)
        self.model = "openai/gpt-oss-120b"
        self.max_content_length = 2000 # Limited content per source

    async def synthesize_response(self, query: str, 
                                  analysis: QueryAnalysis,
                                  web_results: WebSearchResults,
                                  ) -> SynthesizedResponse:
        """Generate comprehensive response from search result"""

        logger.info(f"Synthesizing Response from {web_results.total_results} sources")

        # step 1: prepare and clean search content
        processed_sources = self._process_search_results(web_results.results)

        if not processed_sources:
            logger.warning("No valid source to synthesis from")
            return self._create_fallback_response(query)
        
        # step 2: create synthesis prompt
        synthesis_prompt = self._create_synthesis_prompt(
            query=query,
            analysis=analysis,
            sources=processed_sources
        )

        # setp 3: Generate response using Groq
        try:
            synthesized_content = await self._generate_with_groq(synthesis_prompt)

            # step 4: process and validate response
            response = self._process_synthesized_response(
                content=synthesized_content,
                sources=processed_sources,
                query=query
            )

            logger.info(f"Response synthesized successfully")
            return response
        
        except Exception as e:
            logger.info(f"Synthesis Failed: {e}")
            return self._create_fallback_response(query, str(e))
        
    def _process_search_results(self, results: List[SearchResult]) -> List[Dict[str, Any]]:
        """clean and prepare search results for synthesis"""

        processed = []

        for i, result in enumerate(results[:8]):
            try:
                # clean and truncate content
                content = self._clean_content(result.content)

                if len(content) < 10:   # skip very short content
                    continue

                # truncate if too long
                if len(content) > self.max_content_length:
                    content = content[: self.max_content_length] + "..."

                source = {
                    "id": i + 1,
                    "title": result.title,
                    "url": result.url,
                    "content": content,
                    "score": result.score
                }

                processed.append(source)

            except Exception as e:
                logger.warning(f"Failed to process result: {i}: {e}")
                continue

        logger.info(f"Processed {len(processed)} valid sources")
        return processed
    
    def _clean_content(self, content: str) -> str:
        """Clean and normalize content text"""
        
        # Remove extra whitespace and newlines
        content = re.sub(r'\s+', ' ', content)
        
        # Remove common web artifacts
        content = re.sub(r'(Cookie|Privacy Policy|Terms of Service).*', '', content)
        content = re.sub(r'Advertisement\s*', '', content, flags=re.IGNORECASE)
        
        # Remove HTML-like tags if any
        content = re.sub(r'<[^>]+>', '', content)
        
        return content.strip()
    
    def _create_synthesis_prompt(
        self, 
        query: str, 
        analysis: QueryAnalysis, 
        sources: List[Dict[str, Any]]
    ) -> str:
        """Create perfect Perplexity-style synthesis prompt"""
        
        # Build sources context
        sources_text = ""
        for source in sources:
            sources_text += f"""
    Source [{source['id']}]: {source['title']}
    URL: {source['url']}
    Content: {source['content']}

    ---
    """
        
        prompt = f"""You are Perplexity, a helpful search assistant trained by Perplexity AI.

    Write an accurate, detailed, and comprehensive response to the user's query using the provided search results.

    **User Query**: "{query}"

    **Available Sources**:
    {sources_text}

    **CRITICAL CITATION RULES - READ CAREFULLY**:
    1. You MUST use ONLY square brackets with numbers: [1], [2], [3]
    2. NEVER use 【】 brackets (Chinese/Japanese style)
    3. NEVER use † symbols or other decorations
    4. Place citations at the END of sentences
    5. Every factual claim MUST have a citation

    **CORRECT CITATION EXAMPLES**:
    ✓ "AI can improve healthcare outcomes [1]."
    ✓ "Studies show AI boosts productivity [2]."
    ✓ "Climate models benefit from AI analysis [1][3]."

    **INCORRECT CITATION EXAMPLES** (DO NOT USE):
    ✗ "AI can improve healthcare 【1】"
    ✗ "Studies show AI boosts productivity [1†L5-L8]"
    ✗ "Climate models benefit from AI 【1†source】"

    **FORMATTING REQUIREMENTS**:
    - Use ## for main topics
    - Use ### for subtopics  
    - Use **bold** for key terms
    - Use bullet points (-) for lists
    - Use numbered lists (1.) for steps
    - Cite sources using [1], [2], [3] format ONLY

    Write a comprehensive, well-cited response using ONLY [number] citation format:"""
        
        return prompt

    
    async def _generate_with_groq(self, prompt: str) -> str:
        """Generate response using Groq LLM"""
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert research assistant that creates comprehensive, well-cited responses. Always use proper citations and maintain accuracy."
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for accuracy
                max_tokens=2000,  # Comprehensive responses
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Groq generation failed: {e}")
            raise
    
    def _process_synthesized_response(
        self, 
        content: str, 
        sources: List[Dict[str, Any]], 
        query: str
    ) -> SynthesizedResponse:
        """Process and validate synthesized response"""
        
        # Extract citations from content - handle multiple bracket formats
        citation_patterns = [
            r'\[(\d+)\]',           # Standard: [1]
            r'【(\d+)】',            # CJK brackets: 【1】
            r'\[(\d+)†[^\]]*\]',    # With metadata: [1†source]
            r'【(\d+)†[^】]*】'       # CJK with metadata: 【1†source】
        ]
        
        citations_used = set()
        for pattern in citation_patterns:
            found = re.findall(pattern, content)
            citations_used.update(found)
        
        logger.info(f"Found {len(citations_used)} citations: {citations_used}")
        
        # Create source mapping for citations
        cited_sources = []
        for source in sources:
            if str(source['id']) in citations_used:
                cited_sources.append({
                    "id": source['id'],
                    "title": source['title'],
                    "url": source['url']
                })
        
        # Calculate response metrics
        word_count = len(content.split())
        citation_count = len(citations_used)
        
        logger.info(f"Processed {len(cited_sources)} cited sources from {citation_count} citations")
        
        return SynthesizedResponse(
            query=query,
            response=content,
            sources_used=cited_sources,
            total_sources=len(sources),
            word_count=word_count,
            citation_count=citation_count,
            synthesis_quality_score=self._calculate_quality_score(
                content, citation_count, word_count
            )
        )

    
    def _calculate_quality_score(self, content: str, citations: int, words: int) -> float:
        """Calculate quality score for the synthesized response"""
        
        score = 0.0
        
        # Citation density (good: 1 citation per 50-100 words)
        if words > 0:
            citation_density = citations / (words / 50)
            if 0.5 <= citation_density <= 2.0:
                score += 0.3
            elif citation_density > 0:
                score += 0.1
        
        # Content length (good: 200-800 words for most queries)
        if 200 <= words <= 800:
            score += 0.3
        elif words >= 100:
            score += 0.2
        
        # Structure indicators
        if '##' in content or '###' in content:  # Headers
            score += 0.1
        if '- ' in content or '* ' in content:  # Lists
            score += 0.1
        if citations > 2:  # Multiple sources
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _create_fallback_response(self, query: str, error: str = None) -> SynthesizedResponse:
        """Create fallback response when synthesis fails"""
        
        fallback_content = f"""
            I apologize, but I encountered difficulty synthesizing a comprehensive response for your query: "{query}".
            {f"Error details: {error}" if error else "This may be due to limited search results or processing issues."}
            Please try rephrasing your question or asking about a different topic.
            """
        
        return SynthesizedResponse(
            query=query,
            response=fallback_content,
            sources_used=[],
            total_sources=0,
            word_count=len(fallback_content.split()),
            citation_count=0,
            synthesis_quality_score=0.1
        )









