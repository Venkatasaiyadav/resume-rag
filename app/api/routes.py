"""
ROUTES.PY - FastAPI endpoints for the RAG application

ENDPOINTS:
══════════

POST /ingest          - Upload and index a PDF resume
POST /query           - Ask a question about the resume
POST /query/debug     - Ask with debug info (see chunks, scores)
GET  /health          - Health check
GET  /stats           - Get index statistics
DELETE /index         - Clear all indexed data

TEST WITH POSTMAN:
═════════════════

1. First, POST /ingest with your resume PDF
2. Then, POST /query with your question
3. Use POST /query/debug to see how retrieval works
"""

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import os
import shutil
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.prompt_builder import PromptBuilder
from app.generation.llm_client import LLMClient

router = APIRouter()

# Initialize components (these persist across requests)
pipeline = None
retriever = None
prompt_builder = PromptBuilder()
llm_client = None


def get_pipeline():
    """Lazy initialization of pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = IngestionPipeline()
    return pipeline


def get_retriever():
    """Lazy initialization of retriever"""
    global retriever
    p = get_pipeline()
    if retriever is None:
        retriever = HybridRetriever(
            vector_store=p.vector_store,
            bm25_index=p.bm25_index
        )
    return retriever


def get_llm():
    """Lazy initialization of LLM client"""
    global llm_client
    if llm_client is None:
        llm_client = LLMClient()
    return llm_client


# ═══════════════════════════════════════
# REQUEST/RESPONSE MODELS
# ═══════════════════════════════════════

class QueryRequest(BaseModel):
    """Request body for query endpoint"""
    question: str = Field(..., description="Question about the resume")
    search_mode: str = Field(
        default="hybrid",
        description="Search mode: 'hybrid', 'vector', or 'bm25'"
    )
    top_k: int = Field(
        default=3,
        description="Number of chunks to retrieve"
    )


class QueryResponse(BaseModel):
    """Response from query endpoint"""
    answer: str
    question: str
    search_mode: str
    chunks_used: int


class DebugQueryResponse(BaseModel):
    """Detailed response with debug info"""
    answer: str
    question: str
    search_mode: str
    retrieved_chunks: List[dict]
    prompt_length: int
    model_info: dict


# ═══════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════

@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    Use this to verify the server is running.
    
    POSTMAN: GET http://localhost:8000/health
    """
    return {
        "status": "healthy",
        "message": "RAG Resume API is running!"
    }


@router.post("/ingest")
async def ingest_resume(
    file: UploadFile = File(...),
    force_reindex: bool = False
):
    """
    Upload and index a PDF resume.
    
    POSTMAN SETUP:
    - Method: POST
    - URL: http://localhost:8000/ingest
    - Body: form-data
      - Key: file (type: File)
      - Value: Select your resume.pdf
      - Key: force_reindex (type: Text) 
      - Value: true (optional, to re-index)
    
    WHAT HAPPENS:
    1. Receives PDF file upload
    2. Saves to data/ directory
    3. Runs ingestion pipeline:
       PDF → Extract → Chunk → Embed → Store
    4. Returns statistics about indexing
    """
    # Validate file type
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Save uploaded file
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    print(f"📁 File saved: {file_path}")
    
    # Run ingestion pipeline
    try:
        p = get_pipeline()
        result = p.ingest(file_path, force_reindex=force_reindex)
        
        # Reset retriever to pick up new data
        global retriever
        retriever = None
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_resume(request: QueryRequest):
    """
    Ask a question about the resume.
    
    POSTMAN SETUP:
    - Method: POST
    - URL: http://localhost:8000/query
    - Headers: Content-Type: application/json
    - Body (raw JSON):
      {
        "question": "What are Venkatasai's main technical skills?",
        "search_mode": "hybrid",
        "top_k": 3
      }
    
    WHAT HAPPENS (The Complete RAG Flow):
    1. RETRIEVE: Search indexed chunks using hybrid search
       - BM25 finds keyword matches
       - Vector search finds semantic matches
       - RRF combines both rankings
    2. AUGMENT: Build prompt with retrieved chunks + question
    3. GENERATE: Send prompt to Gemini LLM
    4. Return the answer
    """
    # Validate
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    if request.search_mode not in ["hybrid", "vector", "bm25"]:
        raise HTTPException(
            status_code=400,
            detail="search_mode must be 'hybrid', 'vector', or 'bm25'"
        )
    
    try:
        # Step 1: RETRIEVE relevant chunks
        ret = get_retriever()
        chunks = ret.retrieve(
            query=request.question,
            top_k=request.top_k,
            search_mode=request.search_mode
        )
        
        if not chunks:
            return QueryResponse(
                answer="No relevant information found in the resume for your question.",
                question=request.question,
                search_mode=request.search_mode,
                chunks_used=0
            )
        
        # Step 2: AUGMENT - Build prompt with context
        prompt = prompt_builder.build_prompt(request.question, chunks)
        
        # Step 3: GENERATE - Get answer from LLM
        llm = get_llm()
        answer = llm.generate(prompt)
        
        return QueryResponse(
            answer=answer,
            question=request.question,
            search_mode=request.search_mode,
            chunks_used=len(chunks)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/query/debug", response_model=DebugQueryResponse)
async def query_resume_debug(request: QueryRequest):
    """
    Query with full debug information.
    Shows exactly which chunks were retrieved, their scores, 
    the full prompt, and more.
    
    POSTMAN: Same as /query but use /query/debug URL
    
    Great for understanding HOW the RAG system works!
    You can see:
    - Which chunks were retrieved by BM25 vs Vector
    - RRF scores for each chunk
    - The exact prompt sent to the LLM
    - Which sections of the resume were used
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        # Retrieve
        ret = get_retriever()
        chunks = ret.retrieve(
            query=request.question,
            top_k=request.top_k,
            search_mode=request.search_mode
        )
        
        # Build prompt with debug info
        prompt_data = prompt_builder.build_debug_prompt(request.question, chunks)
        
        # Generate
        llm = get_llm()
        llm_response = llm.generate_with_metadata(prompt_data["prompt"])
        
        return DebugQueryResponse(
            answer=llm_response["answer"],
            question=request.question,
            search_mode=request.search_mode,
            retrieved_chunks=prompt_data["debug"]["chunks_detail"],
            prompt_length=prompt_data["debug"]["prompt_length"],
            model_info={
                "model": llm_response["model"],
                "finish_reason": llm_response["finish_reason"],
                "approx_prompt_tokens": llm_response["prompt_tokens"],
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug query failed: {str(e)}")


@router.get("/stats")
async def get_stats():
    """
    Get statistics about the indexed data.
    
    POSTMAN: GET http://localhost:8000/stats
    """
    try:
        p = get_pipeline()
        return {
            "vector_store_count": p.vector_store.get_count(),
            "bm25_indexed": p.bm25_index._is_built,
            "bm25_chunk_count": len(p.bm25_index.chunks) if p.bm25_index._is_built else 0,
            "embedding_model": p.vector_store.embedding_model.model.get_sentence_embedding_dimension(),
        }
    except Exception as e:
        return {"error": str(e)}


@router.delete("/index")
async def clear_index():
    """
    Clear all indexed data. Use before re-indexing.
    
    POSTMAN: DELETE http://localhost:8000/index
    """
    try:
        p = get_pipeline()
        p.vector_store.clear()
        p.bm25_index = type(p.bm25_index)()  # Reset BM25
        
        global retriever
        retriever = None
        
        return {"status": "success", "message": "All indexed data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))