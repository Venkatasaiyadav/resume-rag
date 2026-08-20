"""
PIPELINE.PY - Complete ingestion pipeline

This orchestrates the entire process of:
1. Reading the PDF
2. Chunking the text
3. Storing in Vector DB (ChromaDB)
4. Building BM25 index

RUN THIS ONCE to index your resume.
After that, the data persists and you can query anytime.
"""

import os
from typing import Optional
from app.ingestion.pdf_extractor import PDFExtractor
from app.ingestion.chunker import ResumeChunker, Chunk
from app.indexing.vector_store import VectorStore
from app.indexing.bm25_index import BM25Index


class IngestionPipeline:
    """
    Complete pipeline: PDF → Chunks → Vector Store + BM25 Index
    """
    
    def __init__(self):
        self.extractor = PDFExtractor()
        self.chunker = ResumeChunker()
        self.vector_store = VectorStore()
        self.bm25_index = BM25Index()
        # Restore persisted BM25 index so retrieval works across restarts.
        # The pickle file only exists locally (storage/ is gitignored), so on
        # fresh deployments we rebuild the index from the Qdrant payloads.
        if self.vector_store.get_count() > 0 and not self.bm25_index.load_index():
            print("🚧 BM25 pickle not found - rebuilding index from Qdrant collection...")
            self.bm25_index.build_index(self.vector_store.get_all_chunks())
            self.bm25_index.save_index()
    
    def ingest(self, pdf_path: str, force_reindex: bool = False) -> dict:
        """
        Ingest a PDF resume into the RAG system.
        
        COMPLETE FLOW:
        ══════════════
        
        Step 1: Extract Text
          resume.pdf → "Udatha Venkatasai\nChennai..."
          
        Step 2: Chunk Text
          "Udatha Venkatasai..." → [Chunk1, Chunk2, ..., ChunkN]
          Each chunk has: text + metadata (section, source, etc.)
          
        Step 3: Index in Vector Store
          For each chunk:
            text → embedding model → [0.23, -0.45, ...] → ChromaDB
          
        Step 4: Build BM25 Index
          For each chunk:
            text → tokenize → ["java", "spring", ...] → BM25 index
        
        Args:
            pdf_path: Path to PDF file
            force_reindex: If True, clear existing data and re-index
            
        Returns:
            Dictionary with ingestion statistics
        """
        print("\n" + "="*60)
        print("🚀 STARTING INGESTION PIPELINE")
        print("="*60)
        
        # Check if already indexed
        if not force_reindex and self.vector_store.get_count() > 0:
            # Try to load BM25 index
            bm25_loaded = self.bm25_index.load_index()
            if bm25_loaded:
                print("✅ Data already indexed! Use force_reindex=True to re-index.")
                return {
                    "status": "already_indexed",
                    "chunk_count": self.vector_store.get_count(),
                    "message": "Data already exists. Use force_reindex=True to re-index."
                }
        
        # Validate file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        # Step 1: Extract text from PDF
        print("\n📄 STEP 1: Extracting text from PDF...")
        raw_text = self.extractor.extract_text(pdf_path)
        
        if not raw_text or len(raw_text) < 50:
            raise ValueError("Extracted text is too short. PDF might be image-based.")
        
        # Step 2: Chunk the text
        print("\n🔪 STEP 2: Chunking text...")
        source_name = os.path.basename(pdf_path)
        chunks = self.chunker.chunk_resume(raw_text, source=source_name)
        
        if not chunks:
            raise ValueError("No chunks created. Check chunking logic.")
        
        # Step 3: Clear old data and add to Vector Store
        print("\n💾 STEP 3: Indexing in Vector Store (ChromaDB)...")
        if force_reindex:
            self.vector_store.clear()
        self.vector_store.add_chunks(chunks)
        
        # Step 4: Build BM25 Index
        print("\n📇 STEP 4: Building BM25 Index...")
        self.bm25_index.build_index(chunks)
        self.bm25_index.save_index()
        
        # Summary
        stats = {
            "status": "success",
            "pdf_path": pdf_path,
            "raw_text_length": len(raw_text),
            "chunk_count": len(chunks),
            "vector_store_count": self.vector_store.get_count(),
            "chunks_detail": [
                {
                    "id": c.chunk_id,
                    "section": c.metadata["section"],
                    "char_count": c.metadata["char_count"],
                    "preview": c.text[:100] + "..."
                }
                for c in chunks
            ]
        }
        
        print("\n" + "="*60)
        print("✅ INGESTION COMPLETE!")
        print(f"  📊 Text extracted: {stats['raw_text_length']} characters")
        print(f"  📦 Chunks created: {stats['chunk_count']}")
        print(f"  💾 Vector store: {stats['vector_store_count']} documents")
        print("="*60)
        
        return stats