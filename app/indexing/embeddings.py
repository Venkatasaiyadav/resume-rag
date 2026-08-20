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

MEMORY STRATEGY (for Render Free / 512 MB):
  - sentence_transformers / torch are imported ONLY inside _get_model(),
    so importing this module never loads PyTorch at startup.
  - The SentenceTransformer model itself is loaded lazily on the FIRST
    embedding call and cached (Singleton) for the life of the process.
  - Model is forced to CPU and limited to a single thread to keep the
    process footprint as small as possible.
"""

import os
from typing import List
import numpy as np
from app.config import settings


class EmbeddingModel:
    """
    Generates text embeddings using Sentence-BERT.

    ARCHITECTURE:
    Text → BERT Tokenizer → BERT Model → Pooling → 384-dim vector

    The model is loaded lazily (on first embedding) and reused.
    """

    _instance = None  # Singleton - only load model once

    def __new__(cls):
        """
        Singleton pattern - ensures the loaded model is shared.
        Creating the instance is CHEAP; the actual SentenceTransformer
        model is only loaded on the first call that needs embeddings.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        print(
            f"🧠 EmbeddingModel ready (lazy) - will load "
            f"{settings.EMBEDDING_MODEL} on first embedding"
        )
        self._initialized = True

    def _get_model(self):
        """
        Lazily load the SentenceTransformer model on first use.

        - Imports sentence_transformers/torch here (inside the method),
          so FastAPI startup and module import never pay the PyTorch
          memory cost.
        - Uses device="cpu" explicitly.
        - Caps PyTorch to a single thread to minimize resident memory.
        """
        if getattr(self, "model", None) is None:
            # Must be set before 'tokenizers' is imported to avoid
            # spawning parallel worker processes on HF tokenizers.
            os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

            print(f"🧠 Loading embedding model: {settings.EMBEDDING_MODEL} "
                  f"(CPU, lazy load)")

            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(settings.EMBEDDING_MODEL, device="cpu")

            # Conservative CPU limits: single thread keeps per-thread
            # allocator arenas small (does not change model output).
            try:
                import torch
                torch.set_num_threads(1)
            except Exception:
                pass

            try:
                self.dimension = model.get_embedding_dimension()
            except Exception:
                self.dimension = model.get_sentence_embedding_dimension()

            print(f"  ✅ Model loaded. Dimension: {self.dimension}")
            self.model = model

        return self.model

    def embed_text(self, text: str) -> List[float]:
        """
        Convert a single text to an embedding vector.

        Returns a List of floats (384 numbers) exactly like before.
        """
        model = self._get_model()
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return embedding.tolist()

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Convert multiple texts to embeddings (batched, low memory).

        Uses batch_size=8 (conservative) and no progress bar.
        """
        model = self._get_model()
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=8,
        )
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