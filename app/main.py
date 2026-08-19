"""
MAIN.PY - FastAPI Application Entry Point

This is where everything comes together.
Run this to start the server.

COMMAND: uvicorn app.main:app --reload --port 8000
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.routes import router

# Create FastAPI app
app = FastAPI(
    title="Resume RAG API",
    description="""
    ## 🤖 RAG-powered Resume Q&A System
    
    Ask questions about Udatha Venkatasai's resume and get 
    intelligent answers powered by:
    
    - **Hybrid Search**: BM25 (keyword) + Vector (semantic)
    - **RRF**: Reciprocal Rank Fusion for combining search results
    - **Gemini LLM**: For generating natural language answers
    
    ### How to use:
    1. **POST /ingest** - Upload resume PDF first
    2. **POST /query** - Ask questions
    3. **POST /query/debug** - Ask with debug info
    """,
    version="1.0.0",
)

# CORS middleware (needed if you add frontend later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """
    Runs when the server starts.
    Loads any pre-existing indexes.
    """
    print("\n" + "="*60)
    print("🚀 RESUME RAG API STARTING UP")
    print("="*60)
    print("📖 Docs: http://localhost:8000/docs")
    print("📮 Postman: Use the endpoints listed in /docs")
    print("="*60 + "\n")


# Run with: uvicorn app.main:app --reload --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)