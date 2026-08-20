"""
VECTOR_STORE.PY - Qdrant Vector Store

WHAT IS QDRANT?
═══════════════

Qdrant is a production-grade vector database written in Rust.
Unlike ChromaDB (file-based), Qdrant runs as a SERVICE (like MySQL/Postgres).

ARCHITECTURE:
  Your App (Python) ──HTTP/gRPC──> Qdrant Server (Docker/Cloud)
                                    - Stores vectors
                                    - Handles similarity search
                                    - Returns nearest neighbors

QDRANT CONCEPTS:
═══════════════

1. COLLECTION: Like a "table" - stores vectors of same dimension
   Example: "resume_chunks" collection with 384-dim vectors

2. POINT: A single entry in a collection
   Structure: {
     id: unique_identifier,
     vector: [0.23, -0.45, ...],  // 384 numbers
     payload: {text: "...", section: "..."}  // metadata
   }

3. DISTANCE METRIC: How to measure similarity
   - Cosine: Best for text (what we use)
   - Euclidean: Physical distances
   - Dot Product: When vectors are normalized

4. SEARCH: Find K nearest points to a query vector
   Returns: List of points with similarity scores

WHY QDRANT FOR PRODUCTION?
- Fast (written in Rust, uses HNSW algorithm)
- Scalable (billions of vectors)
- Cloud-ready (Qdrant Cloud has free tier)
- Docker-friendly (you're already using this!)
- Better filtering than ChromaDB
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, 
    VectorParams, 
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from typing import List, Dict, Optional
import uuid
import hashlib
from app.config import settings
from app.ingestion.chunker import Chunk
from app.indexing.embeddings import EmbeddingModel


class VectorStore:
    """
    Manages Qdrant collection for storing and searching resume chunks.
    """
    
    def __init__(self):
        """
        Initialize Qdrant client.
        
        Connection modes:
        1. Local Docker: http://localhost:6333
        2. Qdrant Cloud: https://xxx.qdrant.io + API key
        3. In-memory: :memory: (for testing)
        """
        print("💾 Initializing Qdrant vector store...")
        print(f"  🔗 Connecting to: {settings.QDRANT_URL}")
        print(f"  🔑 QDRANT_API_KEY configured: {bool(settings.QDRANT_API_KEY)}")
        print(f"  📦 Collection: {settings.COLLECTION_NAME}")
        
        # Create Qdrant client
        if settings.QDRANT_API_KEY:
            # Cloud/authenticated connection
            self.client = QdrantClient(
                url=settings.QDRANT_URL,
                api_key=settings.QDRANT_API_KEY,
            )
        else:
            # Local Docker (no auth)
            self.client = QdrantClient(url=settings.QDRANT_URL)
        
        self.collection_name = settings.COLLECTION_NAME
        self._embedding_model = None  # Lazy - NOT loaded during startup
        
        # Ensure collection exists
        self._ensure_collection()
        
        print(f"  ✅ Collection '{self.collection_name}' ready")
        print(f"  📊 Current point count: {self.get_count()}")

    @property
    def embedding_model(self):
        """
        Lazily create the EmbeddingModel on first embedding call.
        
        Qdrant connection, health checks, get_count(), and stats never
        need the SentenceTransformer model, so it stays unloaded until
        an actual embedding operation (add_chunks / search) runs.
        """
        if self._embedding_model is None:
            self._embedding_model = EmbeddingModel()
        return self._embedding_model
    
    def _ensure_collection(self):
        """
        Create collection if it doesn't exist.
        
        A collection needs:
        - Name (like a table name)
        - Vector configuration (dimension, distance metric)
        """
        # Check if collection exists
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)
        
        if not exists:
            print(f"  📦 Creating collection '{self.collection_name}'...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=settings.EMBEDDING_DIMENSION,  # 384 for all-MiniLM-L6-v2
                    distance=Distance.COSINE,  # Best for text similarity
                ),
            )
            print(f"  ✅ Collection created")
        else:
            print(f"  ✅ Collection already exists")
    
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """
        Add chunks to Qdrant.
        
        FLOW:
        1. Extract texts and generate embeddings
        2. Create Point objects (id, vector, payload)
        3. Upsert to Qdrant (insert or update)
        
        QDRANT POINT STRUCTURE:
        {
            "id": 12345 (int) or "uuid-string",
            "vector": [0.23, -0.45, ...],  // 384 floats
            "payload": {
                "text": "...",
                "section": "Technical Skills",
                "source": "resume.pdf",
                "chunk_id": "resume.pdf_0"  // original chunk_id
            }
        }
        """
        if not chunks:
            print("⚠️ No chunks to add")
            return
        
        print(f"📥 Adding {len(chunks)} chunks to Qdrant...")
        
        # Generate embeddings for all chunks
        print("  🧠 Generating embeddings...")
        texts = [chunk.text for chunk in chunks]
        embeddings = self.embedding_model.embed_texts(texts)
        
        # Create Qdrant points
        # Note: Qdrant IDs must be int or UUID
        # We convert chunk_id to UUID for consistency
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            # Generate a deterministic UUID from chunk_id
            # This ensures same chunk_id always gets same UUID
            point_id = str(uuid.UUID(hashlib.md5(chunk.chunk_id.encode()).hexdigest()))
            
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": chunk.text,
                    "chunk_id": chunk.chunk_id,  # Original ID for reference
                    **chunk.metadata,  # Section, source, etc.
                }
            ))
        
        # Upsert to Qdrant (insert or update)
        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,  # Wait for operation to complete
        )
        
        print(f"  ✅ Added {len(chunks)} chunks. Total: {self.get_count()}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search for similar chunks using vector similarity.
        
        FLOW:
        1. Convert query to embedding
        2. Qdrant finds K nearest vectors
        3. Return formatted results
        
        QDRANT SEARCH RETURNS:
        [
            ScoredPoint(
                id="uuid...",
                score=0.87,  // Cosine similarity (higher = more similar)
                payload={"text": "...", "section": "..."}
            ),
            ...
        ]
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Generate query embedding
        query_embedding = self.embedding_model.embed_text(query)
        
        # Search Qdrant
        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,  # Include metadata
            with_vectors=False,  # Don't return vectors (save bandwidth)
        )
        
        # Format results to match our standard format
        formatted = []
        for result in search_results:
            formatted.append({
                "id": result.payload.get("chunk_id", str(result.id)),
                "text": result.payload.get("text", ""),
                "metadata": {
                    "section": result.payload.get("section", "Unknown"),
                    "source": result.payload.get("source", "Unknown"),
                    "chunk_index": result.payload.get("chunk_index", 0),
                    "char_count": result.payload.get("char_count", 0),
                },
                "score": float(result.score),  # Cosine similarity
            })
        
        return formatted
    
    def clear(self) -> None:
        """Delete the entire collection and recreate it"""
        print(f"🗑️ Clearing collection '{self.collection_name}'...")
        
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print("  ✅ Collection deleted")
        except Exception as e:
            print(f"  ⚠️ Could not delete: {e}")
        
        # Recreate empty collection
        self._ensure_collection()
    
    def get_count(self) -> int:
        """Get total number of points in the collection"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return info.points_count or 0
        except Exception as e:
            print(f"  ⚠️ Could not get count: {e}")
            return 0
    
    def get_collection_info(self) -> Dict:
        """Get detailed info about the collection (for debugging)"""
        try:
            info = self.client.get_collection(collection_name=self.collection_name)
            return {
                "name": self.collection_name,
                "points_count": info.points_count,
                "vectors_config": {
                    "size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance),
                },
                "status": str(info.status),
            }
        except Exception as e:
            return {"error": str(e)}


# Test standalone
if __name__ == "__main__":
    store = VectorStore()
    print("\nCollection Info:", store.get_collection_info())