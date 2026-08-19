"""
EMBEDDINGS.PY - Convert text to numerical vectors

WHAT ARE EMBEDDINGS?
════════════════════

Text is strings. Computers compare numbers better.
Embeddings convert text to numbers (vectors) that capture MEANING.

"Java developer" → [0.23, -0.45, 0.67, 0.12, ..., 0.89]  (384 numbers)
"Spring Boot engineer" → [0.21, -0.43, 0.65, 0.14, ..., 0.87]  (384 numbers)
"I like pizza" → [0.91, 0.12, -0.34, 0.56, ..., 0.23]  (384 numbers)

Notice: "Java developer" and "Spring Boot engineer" have SIMILAR numbers
because they're semantically related!
"I like pizza" has DIFFERENT numbers because it's unrelated.

HOW SIMILARITY WORKS:
  Cosine Similarity = dot(A, B) / (|A| * |B|)
  
  cos_sim("Java developer", "Spring Boot engineer") ≈ 0.85 (HIGH - similar!)
  cos_sim("Java developer", "I like pizza") ≈ 0.12 (LOW - different!)

MODEL: all-MiniLM-L6-v2
  - 384 dimensions (each text becomes 384 numbers)
  - Fast inference
  - Good quality for English text
  - ~80MB model size
  - Trained on 1B+ sentence pairs
"""

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
from app.config import settings


class EmbeddingModel:
    """
    Generates text embeddings using Sentence-BERT.
    
    ARCHITECTURE:
    Text → BERT Tokenizer → BERT Model → Pooling → 384-dim vector
    
    The model is loaded ONCE and reused for all embeddings.
    """
    
    _instance = None  # Singleton - only load model once
    
    def __new__(cls):
        """
        Singleton pattern - ensures model is loaded only ONCE.
        Loading the model takes ~5 seconds.
        We don't want to reload it for every request.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        print(f"🧠 Loading embedding model: {settings.EMBEDDING_MODEL}")
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = self.model.get_sentence_embedding_dimension()
        print(f"  ✅ Model loaded. Dimension: {self.dimension}")
        self._initialized = True
    
    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text to embedding vector.
        
        Args:
            text: Input text string
            
        Returns:
            List of floats (384 numbers)
            
        Example:
            embed_text("Java developer") 
            → [0.23, -0.45, 0.67, ..., 0.89]  # 384 floats
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (batched for speed).
        
        WHY BATCH?
        Individual: 10 texts × 50ms = 500ms
        Batched:    10 texts × 1 batch = 80ms (GPU parallelism)
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, 
                                        show_progress_bar=True,
                                        batch_size=32)
        return embeddings.tolist()
    
    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two texts.
        Useful for debugging/testing.
        
        Returns: Float between -1 and 1 (1 = identical meaning)
        """
        emb1 = np.array(self.embed_text(text1))
        emb2 = np.array(self.embed_text(text2))
        
        cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        return float(cosine_sim)


# Test standalone
if __name__ == "__main__":
    model = EmbeddingModel()
    
    # Test embedding
    emb = model.embed_text("Java Spring Boot developer")
    print(f"Embedding dimension: {len(emb)}")
    print(f"First 5 values: {emb[:5]}")
    
    # Test similarity
    sim1 = model.similarity("Java developer", "Spring Boot engineer")
    sim2 = model.similarity("Java developer", "I like pizza")
    print(f"\nSimilarity (Java dev ↔ Spring Boot eng): {sim1:.4f}")
    print(f"Similarity (Java dev ↔ I like pizza):     {sim2:.4f}")