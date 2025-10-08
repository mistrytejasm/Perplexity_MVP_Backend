from fastapi import FastAPI, UploadFile, File, BackgroundTasks
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

app = FastAPI(title="Perplexity_MVP_Backend")

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

# -------------- Search Orchestrator --------------------
from services.search_orchestrator import SearchOrchestrator
from models.schemas import SearchRequest, SearchResponse
from datetime import datetime

@app.post("/search-orchestrator")
async def search_orchestrator(request: SearchRequest):
    """
    Main Search Endpoint that executes the full pipeline:
    1. Analyze query with groq
    2. search web with Tavily
    3. Synthesize response
    """
    logger.info(f"Search Request Recived: {request.query}")
    search_orchestrator = SearchOrchestrator()

    try:
        # Execute the full search pipeline
        response = await search_orchestrator.execute_search(request)
        logger.info(f"Search Completed Successfully")
        return response
    
    except Exception as e:
        logger.error(f"Search endpoint failed: {e}")
        return SearchResponse(
            original_query=request.query,
            analusis=None,
            web_results=None,
            synthesized_response=None,
            status=f"error: {str(e)}",
            timestemp=datetime.now().isoformate()
        )


# ---- Document (conetnt Extraction file)) ----------------
from services.document.content_extractor import DocumentContentExtractor
from pathlib import Path
import os
import uuid

# create uploaded directory
UPLOAD_DIR = Path("temp/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def cleanup_file(file_path: str, delay: int = 300):
    """Delete file after specific delay(default 5 minutes)"""
    async def delayed_delete():
        await asyncio.sleep(delay)
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Auto-cleaned Up: {file_path}")

    asyncio.create_task(delayed_delete())

@app.post("/document-extraction")
async def document_extraction(
    file: UploadFile = File(...), 
    background_tasks: BackgroundTasks = BackgroundTasks()  
):
    """Extract text content from uploaded PDF file"""
    
    logger.info(f"Received file: {file.filename}")
    
    # Validate file type
    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF files are supported"}
    
    # Generate unique filename (add timestamp to avoid conflicts)
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_filename = f"{file.filename}"
    file_path = UPLOAD_DIR / unique_filename
    
    try:
        # Save uploaded file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        logger.info(f"File saved to: {file_path}")
        
        # Extract content
        extractor = DocumentContentExtractor()
        result = extractor.extract_from_pdf(file_path=str(file_path))
        
        logger.info(f"Extraction complete: {result.total_pages} pages")
        
        # Schedule cleanup after processing
        background_tasks.add_task(
            lambda: os.remove(file_path) if os.path.exists(file_path) else None
        )
        
        return {
            "success": True,
            "filename": file.filename,
            "text": result,            
            "total_pages": result.total_pages,
            "extraction_method": result.extraction_method,
            "total_characters": len(result.text)
        }
    
    except Exception as e:
        logger.error(f"Document extraction failed: {e}")
        
        # Immediate cleanup on error
        if file_path.exists():
            os.remove(file_path)
        
        return {"error": str(e), "filename": file.filename}

# --- Chunking Text Data service --------
from services.document.chunking_service import DocumentChunkingService
import uuid

chunking_services = DocumentChunkingService()

@app.post("/chunking-text-service")
async def chunking_text_service(text: str, document_id: str = None):
    """
    Chunk text into smaller semantic pieces for RAG Processing
    """

    try:
        if not document_id:
            document_id = f"doc_{uuid.uuid4().hex[:8]}"

        # chunk the text
        chunks = chunking_services.chunk_document(text=text, document_id=document_id)

        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        logger.info(f"Content: {chunks}")

        return {
            "success": True,
            "document_id": document_id,
            "total_chunks": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "content_preview": chunk.content
                }
                for chunk in chunks
            ],
            "statistics": {
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "avg_tokens_per_chunk": sum(chunk.token_count for chunk in chunks) // len(chunks) if chunks else 0,
                "pages_processed": max(chunk.page_number for chunk in chunks) if chunks else 0
            }
        }
    
    except Exception as e:
        logger.error(f"❌ Chunking failed: {e}")
        return {"error": str(e)}
            
# --- Combined Endpoint: Extract + Chunk ---
from pathlib import Path
from datetime import datetime
import os
from services.document.content_extractor import DocumentContentExtractor

@app.post("/extract-and-chunk")
async def extract_and_chunk(file: UploadFile = File(...)):
    """
    Upload PDF, extract Text, and chunk it in one go
    """
    logger.info(f"Extract & Chunk request for: {file.filename}")

    if not file.filename.endswith('.pdf'):
        return {"error": "Only PDF Files are Supported"}

    # Step 1: Upload File
    # save file temporarily
    UPLOAD_DIR = Path("temp/uploads")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    unique_filename = f"{file.filename}"
    file_path = UPLOAD_DIR / unique_filename

    try:
        # save file
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File Saved: {file_path}")

        # Extracted Text
        extractor = DocumentContentExtractor()
        extraction_result = extractor.extract_from_pdf(file_path=str(file_path))

        logger.info(f"Extracted {extraction_result.total_pages} pages")
        # logger.info(f"Extracted_result Text: {extraction_result}")

        # generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:8]}"

        # chunk the extracted text
        chunks = chunking_services.chunk_document(
            text = extraction_result.text,
            document_id = document_id
        )

        logger.info(f"Created {len(chunks)} chunks")

        return {
            "success": True,
            "filename": file.filename,
            "document_id": document_id,
            "extraction": {
                "total_pages": extraction_result.total_pages,
                "extraction_method": extraction_result.extraction_method,
                "total_characters": len(extraction_result.text)
            },
            "chunking": {
                "total_chunks": len(chunks),
                "total_tokens": sum(chunk.token_count for chunk in chunks),
                "avg_tokens_per_chunk": sum(chunk.token_count for chunk in chunks) // len(chunks)
            },
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "content_preview": chunk.content
                }
                # for chunk in chunks[:5]  # Return first 5 chunks as preview
                for chunk in chunks
            ]
        }
    
    except Exception as e:
        logger.error(f"Extract & Chunk failed: {e}")
        return {"error": str(e)}
    
    finally:
        # Cleanup
        if file_path.exists():
            os.remove(file_path)

# -------------- store chunks in database -------------------



if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
