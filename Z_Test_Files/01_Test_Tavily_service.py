from services.tavily_service import TavilyService
import asyncio
from dotenv import load_dotenv
from config.settings import settings
from fastapi import FastAPI
from logger_config import logging

logger = logging.getLogger().setLevel(logging.INFO)
load_dotenv()

async def test_tavily():
    search = TavilyService()
    search_terms = ["what is AI?", "how AI works?"]
    result = await search.search_multiple(search_terms)
    print(result)
   
if __name__ == "__main__":
    asyncio.run(test_tavily())

# from tavily import TavilyClient
# tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)
# response = tavily_client.search("Who is Leo Messi?")
# print(response)
