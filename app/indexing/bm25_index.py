"""
BM25_INDEX.PY - Keyword-based search using BM25

WHAT IS BM25?
═════════════

BM25 = Best Match 25. A ranking function used by search engines.

HOW IT WORKS (Simplified):
  1. TOKENIZE: Split text into words
     "Java Spring Boot" → ["java", "spring", "boot"]
  
  2. For each query word, calculate score for each document:
  
     score = IDF(word) × TF_component
     
     IDF (Inverse Document Frequency):
       = How RARE is this word across all documents?
       - "the" appears in every doc → IDF ≈ 0 (not useful)
       - "Kafka" appears in 1 doc → IDF is HIGH (very useful!)
       
     TF (Term Frequency) component:
       = How often does this word appear in THIS document?
       - With saturation: appearing 10x isn't 10× better than 1x
       - With length normalization: longer docs don't unfairly win

  3. Sum scores for all query words → final document score

EXAMPLE:
  Query: "Spring Boot REST API"
  
  Chunk A: "Built REST APIs using Spring Boot and MongoDB"
    - "spring": TF=1, IDF=medium → score: 2.3
    - "boot": TF=1, IDF=medium → score: 2.1
    - "rest": TF=1, IDF=high → score: 3.5
    - "api": TF=1, IDF=low → score: 1.2
    - Total: 9.1 ✓ HIGH
  
  Chunk B: "Studied at Dr MGR university in Chennai"
    - "spring": TF=0 → score: 0
    - "boot": TF=0 → score: 0
    - "rest": TF=0 → score: 0
    - "api": TF=0 → score: 0
    - Total: 0.0 ✗ LOW

BM25 vs Vector Search:
  BM25 WINS when: exact keywords matter ("ChromaDB", "Kafka", "Redis")
  Vector WINS when: meaning matters ("backend frameworks" → finds "Spring Boot")
  HYBRID WINS always: combine both! (That's our approach)
"""

from rank_bm25 import BM25Okapi
from typing import List, Dict, Optional, Tuple
import re
import pickle
import os
from app.ingestion.chunker import Chunk
from app.config import settings


class BM25Index:
    """
    BM25 keyword search index for resume chunks.
    
    Stores tokenized documents and provides keyword-based search.
    """
    
    def __init__(self):
        self.bm25 = None
        self.chunks: List[Chunk] = []
        self.tokenized_corpus: List[List[str]] = []
        self._is_built = False
    
    def build_index(self, chunks: List[Chunk]) -> None:
        """
        Build BM25 index from chunks.
        
        FLOW:
        1. Store chunks
        2. Tokenize each chunk's text
        3. Build BM25 index from tokenized corpus
        
        TOKENIZATION:
        "Built REST APIs using Spring Boot" 
        → ["built", "rest", "apis", "using", "spring", "boot"]
        
        Note: We lowercase and remove punctuation for better matching.
        "Spring" should match "spring" in query.
        """
        print(f"📇 Building BM25 index with {len(chunks)} chunks...")
        
        self.chunks = chunks
        
        # Tokenize all chunks
        self.tokenized_corpus = [
            self._tokenize(chunk.text) for chunk in chunks
        ]
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._is_built = True
        
        # Print some stats
        vocab = set()
        for tokens in self.tokenized_corpus:
            vocab.update(tokens)
        
        print(f"  ✅ BM25 index built")
        print(f"  📊 Vocabulary size: {len(vocab)} unique words")
        print(f"  📊 Avg tokens per chunk: {sum(len(t) for t in self.tokenized_corpus) / len(self.tokenized_corpus):.0f}")
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Search using BM25 keyword matching.
        
        FLOW:
        1. Tokenize the query
        2. BM25 scores each document against query tokens
        3. Return top_k highest scoring documents
        
        Args:
            query: User's question
            top_k: Number of results
            
        Returns:
            List of dicts with text, metadata, score
        """
        if not self._is_built:
            raise ValueError("BM25 index not built yet! Call build_index() first.")
        
        top_k = top_k or settings.TOP_K_RESULTS
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top_k indices (sorted by score, descending)
        top_indices = sorted(range(len(scores)), 
                           key=lambda i: scores[i], 
                           reverse=True)[:top_k]
        
        # Format results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some match
                results.append({
                    "id": self.chunks[idx].chunk_id,
                    "text": self.chunks[idx].text,
                    "metadata": self.chunks[idx].metadata,
                    "score": float(scores[idx]),
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25.
        
        Steps:
        1. Lowercase
        2. Remove special characters (keep alphanumeric)
        3. Split by whitespace
        4. Remove very short tokens (single chars except meaningful ones)
        5. Remove common stop words
        
        "Built REST APIs using Spring Boot, and MongoDB."
        → ["built", "rest", "apis", "using", "spring", "boot", "mongodb"]
        """
        # Lowercase
        text = text.lower()
        
        # Remove special characters, keep alphanumeric and spaces
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Split by whitespace
        tokens = text.split()
        
        # Remove stop words and very short tokens
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at',
            'to', 'for', 'of', 'with', 'by', 'from', 'is', 'was',
            'are', 'were', 'be', 'been', 'being', 'have', 'has',
            'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'it', 'its', 'he', 'she', 'they', 'them',
        }
        
        tokens = [t for t in tokens if len(t) > 1 and t not in stop_words]
        
        return tokens
    
    def save_index(self, path: str = "storage/bm25_index.pkl") -> None:
        """Save BM25 index to disk for persistence"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'tokenized_corpus': self.tokenized_corpus,
            }, f)
        print(f"💾 BM25 index saved to {path}")
    
    def load_index(self, path: str = "storage/bm25_index.pkl") -> bool:
        """Load BM25 index from disk"""
        if not os.path.exists(path):
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.chunks = data['chunks']
        self.tokenized_corpus = data['tokenized_corpus']
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self._is_built = True
        print(f"📂 BM25 index loaded from {path} ({len(self.chunks)} chunks)")
        return True