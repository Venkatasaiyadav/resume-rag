"""
HYBRID_RETRIEVER.PY - Combines Vector + BM25 search with RRF

THE COMPLETE RETRIEVAL PIPELINE:
═══════════════════════════════

  User Query: "What are Venkatasai's Java projects?"
       │
       ├─────────────────────┬────────────────────────┐
       ▼                     ▼                        │
  ┌──────────────┐    ┌──────────────┐               │
  │ BM25 Search  │    │ Vector Search│               │
  │              │    │              │               │
  │ Tokenize     │    │ Embed query  │               │
  │ query words  │    │ to vector    │               │
  │              │    │              │               │
  │ Match exact  │    │ Find nearest │               │
  │ keywords in  │    │ vectors in   │               │
  │ chunks       │    │ ChromaDB     │               │
  └──────┬───────┘    └──────┬───────┘               │
         │                   │                        │
         ▼                   ▼                        │
  BM25 Results:        Vector Results:                │
  1. Skills chunk      1. Experience chunk             │
  2. Experience chunk  2. Skills chunk                 │
  3. Summary chunk     3. Projects chunk               │
         │                   │                        │
         └─────────┬─────────┘                        │
                   ▼                                  │
            ┌──────────────┐                          │
            │  RRF Fusion  │                          │
            │              │                          │
            │  Combine by  │                          │
            │  rank, not   │                          │
            │  by score    │                          │
            └──────┬───────┘                          │
                   ▼                                  │
            Final Ranked Results                      │
            1. Skills chunk (in both!)                │
            2. Experience chunk (in both!)            │
            3. Summary chunk (BM25 only)              │
            4. Projects chunk (Vector only)           │
"""

from typing import List, Dict, Optional
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index
from app.retrieval.rrf import ReciprocalRankFusion
from app.config import settings


class HybridRetriever:
    """
    Combines BM25 (sparse) and Vector (dense) search using RRF.
    """
    
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        rrf_k: int = None
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.rrf = ReciprocalRankFusion(k=rrf_k)
    
    def retrieve(
        self,
        query: str,
        top_k: int = None,
        search_mode: str = "hybrid"  # "hybrid", "vector", "bm25"
    ) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        
        Args:
            query: User's question
            top_k: Number of results to return
            search_mode: Which search method to use
                - "hybrid": Both BM25 + Vector with RRF (RECOMMENDED)
                - "vector": Only semantic/vector search
                - "bm25": Only keyword/BM25 search
                
        Returns:
            List of relevant chunks with scores
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        print(f"\n🔍 Retrieving for: '{query}' (mode: {search_mode})")
        
        if search_mode == "vector":
            results = self._vector_search(query, top_k)
        elif search_mode == "bm25":
            results = self._bm25_search(query, top_k)
        elif search_mode == "hybrid":
            results = self._hybrid_search(query, top_k)
        else:
            raise ValueError(f"Unknown search mode: {search_mode}")
        
        print(f"  📋 Retrieved {len(results)} chunks")
        return results
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict]:
        """Pure vector/semantic search"""
        print("  🧠 Running vector search...")
        results = self.vector_store.search(query, top_k)
        for i, r in enumerate(results):
            print(f"    {i+1}. [{r['id']}] score={r['score']:.4f} "
                  f"- {r['text'][:60]}...")
        return results
    
    def _bm25_search(self, query: str, top_k: int) -> List[Dict]:
        """Pure BM25/keyword search"""
        print("  📝 Running BM25 search...")
        results = self.bm25_index.search(query, top_k)
        for i, r in enumerate(results):
            print(f"    {i+1}. [{r['id']}] score={r['score']:.4f} "
                  f"- {r['text'][:60]}...")
        return results
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict]:
        """
        Hybrid search: BM25 + Vector combined with RRF.
        
        FLOW:
        1. Run BM25 search → get top results by keyword matching
        2. Run Vector search → get top results by semantic similarity
        3. Feed both result lists to RRF
        4. RRF combines and re-ranks based on position in both lists
        5. Return top_k fused results
        """
        print("  🔄 Running HYBRID search (BM25 + Vector + RRF)...")
        
        # Get more results from each method than final top_k
        # This ensures RRF has enough candidates to work with
        fetch_k = top_k * 2
        
        # Step 1: BM25 Search
        print("  📝 Step 1: BM25 keyword search...")
        bm25_results = self.bm25_index.search(query, fetch_k)
        
        # Step 2: Vector Search
        print("  🧠 Step 2: Vector semantic search...")
        vector_results = self.vector_store.search(query, fetch_k)
        
        # Step 3: RRF Fusion
        print("  🔗 Step 3: RRF Fusion...")
        fused_results = self.rrf.fuse(
            result_lists=[bm25_results, vector_results],
            top_k=top_k
        )
        
        return fused_results