"""
RRF.PY - Reciprocal Rank Fusion

THE FUSION PROBLEM:
══════════════════

We have TWO search methods returning results:

  BM25 Results (keyword match):
    Rank 1: Chunk_A (score: 15.7)
    Rank 2: Chunk_C (score: 12.3)
    Rank 3: Chunk_B (score: 8.1)
    
  Vector Results (semantic match):
    Rank 1: Chunk_B (score: 0.95)
    Rank 2: Chunk_A (score: 0.82)
    Rank 3: Chunk_D (score: 0.71)

PROBLEMS with simple combination:
  1. Different scales: BM25 scores are 0-20+, vector scores are 0-1
  2. Can't just add them: 15.7 + 0.82 = 16.52 (BM25 dominates!)
  3. Can't normalize easily: distributions are different

RRF SOLUTION:
  Ignore scores completely! Use only RANKS.
  
  Formula: RRF_score(doc) = Σ 1/(k + rank_i(doc))
           where k = 60 (constant)
  
  WHY k = 60?
  - Too small (k=1): Top-ranked results dominate too much
    rank 1: 1/(1+1)=0.500, rank 2: 1/(1+2)=0.333 → 50% gap!
  - Too large (k=1000): All ranks become nearly equal
    rank 1: 1/(1000+1)=0.000999, rank 2: 1/(1000+2)=0.000998 → 0.1% gap
  - k=60: Nice balance
    rank 1: 1/(60+1)=0.01639, rank 2: 1/(60+2)=0.01613 → 1.6% gap

FULL EXAMPLE:
═══════════

  BM25 Ranking:    A=1, C=2, B=3
  Vector Ranking:  B=1, A=2, D=3
  
  RRF(A) = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252
  RRF(B) = 1/(60+3) + 1/(60+1) = 0.01587 + 0.01639 = 0.03226
  RRF(C) = 1/(60+2) + 0         = 0.01613             = 0.01613
  RRF(D) = 0        + 1/(60+3)  = 0.01587             = 0.01587
  
  Final Ranking: A (0.0325) > B (0.0323) > C (0.0161) > D (0.0159)
  
  NOTE: A wins because it appeared in BOTH result sets (rank 1 + rank 2)
        C and D appeared in only ONE set, so they score lower.
  
  This is the MAGIC of RRF: it rewards documents found by MULTIPLE methods!
"""

from typing import List, Dict, Tuple
from collections import defaultdict
from app.config import settings


class ReciprocalRankFusion:
    """
    Combines results from multiple search methods using RRF.
    """
    
    def __init__(self, k: int = None):
        """
        Args:
            k: RRF constant (default 60 from original paper)
        """
        self.k = k or settings.RRF_K
    
    def fuse(
        self,
        result_lists: List[List[Dict]],
        top_k: int = None
    ) -> List[Dict]:
        """
        Fuse multiple ranked result lists into one.
        
        Args:
            result_lists: List of result lists from different search methods.
                         Each result is a dict with at least 'id' and 'text'.
                         Example: [bm25_results, vector_results]
            top_k: Number of final results to return
            
        Returns:
            Fused and re-ranked list of results
            
        VISUALIZATION:
        
        Input:
          List 1 (BM25):   [A, C, B, E]
          List 2 (Vector):  [B, A, D, C]
          
        Processing:
          For each document, sum 1/(k + rank) across all lists
          
        Output:
          [A, B, C, D, E]  (re-ranked by RRF score)
        """
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Store RRF scores and document data
        rrf_scores: Dict[str, float] = defaultdict(float)
        doc_data: Dict[str, Dict] = {}  # Store the actual document data
        
        # Process each result list
        for list_idx, results in enumerate(result_lists):
            method_name = ["BM25", "Vector", "Other"][min(list_idx, 2)]
            
            for rank, result in enumerate(results):
                doc_id = result['id']
                
                # RRF formula: 1 / (k + rank + 1)
                # rank is 0-indexed, so rank+1 makes it 1-indexed
                rrf_score = 1.0 / (self.k + rank + 1)
                rrf_scores[doc_id] += rrf_score
                
                # Store document data (first occurrence wins)
                if doc_id not in doc_data:
                    doc_data[doc_id] = result
                
                # Debug info
                print(f"  📊 RRF: {method_name} rank {rank+1} → "
                      f"doc '{doc_id}' gets +{rrf_score:.5f} "
                      f"(total: {rrf_scores[doc_id]:.5f})")
        
        # Sort by RRF score (descending)
        sorted_docs = sorted(rrf_scores.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
        
        # Build final results
        fused_results = []
        for doc_id, rrf_score in sorted_docs[:top_k]:
            result = doc_data[doc_id].copy()
            result['rrf_score'] = rrf_score
            result['original_score'] = result.get('score', 0)
            result['score'] = rrf_score  # Replace with RRF score
            fused_results.append(result)
        
        print(f"\n  🏆 RRF Final Ranking:")
        for i, r in enumerate(fused_results):
            print(f"    {i+1}. [{r['id']}] RRF={r['rrf_score']:.5f} "
                  f"- {r['text'][:60]}...")
        
        return fused_results


# Test standalone
if __name__ == "__main__":
    rrf = ReciprocalRankFusion(k=60)
    
    bm25_results = [
        {"id": "chunk_0", "text": "Java developer with experience...", "score": 15.7},
        {"id": "chunk_2", "text": "Technical Skills: Java, SQL...", "score": 12.3},
        {"id": "chunk_1", "text": "Built REST APIs using Spring...", "score": 8.1},
    ]
    
    vector_results = [
        {"id": "chunk_1", "text": "Built REST APIs using Spring...", "score": 0.95},
        {"id": "chunk_0", "text": "Java developer with experience...", "score": 0.82},
        {"id": "chunk_3", "text": "Education: B.Tech CSE...", "score": 0.71},
    ]
    
    fused = rrf.fuse([bm25_results, vector_results], top_k=3)
    
    print("\n=== FUSED RESULTS ===")
    for r in fused:
        print(f"  {r['id']}: RRF={r['rrf_score']:.5f}")