"""
CONFIG.PY - Central Configuration

Updated to use Qdrant instead of ChromaDB.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """All application settings in one place"""

    # --- API Keys ---
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # --- Embedding Model ---
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "all-MiniLM-L6-v2"
    )
    EMBEDDING_DIMENSION: int = 384

    # --- Qdrant Configuration ---
    QDRANT_URL: str = os.getenv(
        "QDRANT_URL",
        "http://localhost:6333"
    )

    QDRANT_API_KEY: str = os.getenv(
        "QDRANT_API_KEY",
        ""
    )

    COLLECTION_NAME: str = os.getenv(
        "QDRANT_COLLECTION_NAME",
        "resume_chunks"
    )

    # --- Chunking ---
    CHUNK_SIZE: int = int(
        os.getenv("CHUNK_SIZE", "500")
    )

    CHUNK_OVERLAP: int = int(
        os.getenv("CHUNK_OVERLAP", "50")
    )

    # --- Retrieval ---
    TOP_K_RESULTS: int = int(
        os.getenv("TOP_K_RESULTS", "5")
    )

    RRF_K: int = int(
        os.getenv("RRF_K", "60")
    )

    # --- LLM (Groq) ---
    LLM_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "openai/gpt-oss-120b"
    )

    MAX_TOKENS: int = 1024
    TEMPERATURE: float = 0.3


# Create settings object
settings = Settings()


# Debug configuration
print("========== QDRANT CONFIG ==========")
print("QDRANT_URL:", settings.QDRANT_URL)
print(
    "QDRANT_API_KEY configured:",
    bool(settings.QDRANT_API_KEY)
)
print("COLLECTION:", settings.COLLECTION_NAME)
print("===================================")