"""
CHUNKER.PY - Split resume text into meaningful chunks

CHUNKING STRATEGY FOR RESUME:
════════════════════════════

We use a TWO-PHASE approach:

PHASE 1: Section-Based Chunking
    Resume naturally has sections:
    - Professional Summary
    - Professional Experience  
    - Technical Skills
    - Education
    - Technical Projects
    
    We split by these section headers FIRST.
    This preserves the MEANING of each section.

PHASE 2: Size-Based Sub-Chunking
    If any section is too long (>CHUNK_SIZE), we split it further
    using recursive character splitting.
    
    We split on these separators (in order):
    1. Double newline (paragraph break)
    2. Single newline (line break)
    3. Period + space (sentence break)
    4. Space (word break) - last resort

OVERLAP:
    Each chunk overlaps slightly with the next.
    Why? Consider this text split at position 100:
    
    Chunk 1: "...built REST APIs using Spring"
    Chunk 2: "Boot and MongoDB for healthcare..."
    
    Without overlap, searching "Spring Boot" finds NEITHER chunk!
    
    With 50-char overlap:
    Chunk 1: "...built REST APIs using Spring Boot and Mon"
    Chunk 2: "Spring Boot and MongoDB for healthcare..."
    
    Now "Spring Boot" is found in Chunk 2!

METADATA:
    Each chunk gets metadata:
    - section: Which resume section it belongs to
    - chunk_index: Its position in the document
    - source: The source file name
    
    This helps with filtering and debugging.
"""

import re
from typing import List, Dict
from dataclasses import dataclass
from app.config import settings


@dataclass
class Chunk:
    """
    Represents a single chunk of text with metadata.
    
    Attributes:
        text: The actual text content
        metadata: Dictionary with section, index, source info
        chunk_id: Unique identifier for this chunk
    """
    text: str
    metadata: Dict
    chunk_id: str


class ResumeChunker:
    """
    Chunks resume text using section-based + recursive strategy.
    """
    
    # Resume section headers to split on
    # These are the typical sections in YOUR resume
    SECTION_HEADERS = [
        "Professional Summary",
        "Professional Experience",
        "Technical Skills",
        "Education",
        "Technical Projects",
        "Certifications",
        "Achievements",
        "Contact",
    ]
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP
    
    def chunk_resume(self, text: str, source: str = "resume.pdf") -> List[Chunk]:
        """
        Main chunking method.
        
        Flow:
        1. Split text into sections based on headers
        2. For each section, if too long, split further
        3. Add metadata to each chunk
        4. Return list of Chunk objects
        
        Args:
            text: Full resume text
            source: Source filename for metadata
            
        Returns:
            List of Chunk objects with text and metadata
        """
        print("\n🔪 Starting chunking process...")
        
        # Phase 1: Split by sections
        sections = self._split_by_sections(text)
        print(f"  📑 Found {len(sections)} sections")
        
        # Phase 2: Sub-chunk large sections
        chunks = []
        chunk_index = 0
        
        for section_name, section_text in sections:
            if len(section_text.strip()) < 20:  # Skip near-empty sections
                continue
                
            # If section fits in one chunk, keep it as is
            if len(section_text) <= self.chunk_size:
                chunk = Chunk(
                    text=section_text.strip(),
                    metadata={
                        "section": section_name,
                        "source": source,
                        "chunk_index": chunk_index,
                        "char_count": len(section_text.strip()),
                    },
                    chunk_id=f"{source}_{chunk_index}"
                )
                chunks.append(chunk)
                chunk_index += 1
                print(f"  ✅ Section '{section_name}': 1 chunk ({len(section_text)} chars)")
            else:
                # Section too long - split recursively
                sub_texts = self._recursive_split(section_text)
                for sub_text in sub_texts:
                    if len(sub_text.strip()) < 20:
                        continue
                    chunk = Chunk(
                        text=sub_text.strip(),
                        metadata={
                            "section": section_name,
                            "source": source,
                            "chunk_index": chunk_index,
                            "char_count": len(sub_text.strip()),
                        },
                        chunk_id=f"{source}_{chunk_index}"
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                print(f"  ✅ Section '{section_name}': {len(sub_texts)} chunks")
        
        print(f"  📦 Total chunks created: {len(chunks)}")
        
        # Print chunk preview
        for c in chunks:
            print(f"    [{c.chunk_id}] {c.metadata['section']}: "
                  f"{c.text[:80]}...")
        
        return chunks
    
    def _split_by_sections(self, text: str) -> List[tuple]:
        """
        Split text by resume section headers.
        
        HOW IT WORKS:
        1. Find all section header positions in the text
        2. Split text between consecutive headers
        3. Each split becomes (section_name, section_text)
        
        Example:
            "Professional Summary\nJava developer...\nTechnical Skills\nJava, Spring..."
            →
            [("Professional Summary", "Java developer..."),
             ("Technical Skills", "Java, Spring...")]
        """
        sections = []
        
        # Find positions of all section headers
        header_positions = []
        for header in self.SECTION_HEADERS:
            # Case-insensitive search for section headers
            pattern = re.compile(re.escape(header), re.IGNORECASE)
            for match in pattern.finditer(text):
                header_positions.append((match.start(), header))
        
        # Sort by position in text
        header_positions.sort(key=lambda x: x[0])
        
        if not header_positions:
            # No headers found - treat entire text as one section
            return [("Full Resume", text)]
        
        # Handle text before first header (contact info, name)
        if header_positions[0][0] > 0:
            pre_text = text[:header_positions[0][0]]
            if pre_text.strip():
                sections.append(("Contact Information", pre_text))
        
        # Split between consecutive headers
        for i, (pos, header) in enumerate(header_positions):
            # End position is start of next header (or end of text)
            if i + 1 < len(header_positions):
                end_pos = header_positions[i + 1][0]
            else:
                end_pos = len(text)
            
            section_text = text[pos:end_pos]
            # Remove the header itself from the text but keep it for context
            sections.append((header, section_text))
        
        return sections
    
    def _recursive_split(self, text: str) -> List[str]:
        """
        Recursively split text into smaller chunks.
        
        SPLITTING HIERARCHY (tries in order):
        1. "\n\n" - Paragraph breaks (best - preserves complete thoughts)
        2. "\n"   - Line breaks (good - preserves bullet points)
        3. ". "   - Sentence breaks (ok - may split related sentences)
        4. " "    - Word breaks (last resort)
        
        WITH OVERLAP:
        Each chunk includes the last `chunk_overlap` characters 
        from the previous chunk.
        
        Example with chunk_size=100, overlap=20:
        Text: "AAAA....(100 chars)....BBBB....(100 chars)....CCCC"
        Chunk 1: "AAAA....(100 chars)..."
        Chunk 2: "...(last 20 of chunk1)...BBBB....(80 chars)..."
        """
        separators = ["\n\n", "\n", ". ", " "]
        
        return self._split_with_separators(text, separators)
    
    def _split_with_separators(self, text: str, separators: List[str]) -> List[str]:
        """Split text using the first working separator"""
        if len(text) <= self.chunk_size:
            return [text]
        
        # Try each separator
        for separator in separators:
            if separator in text:
                parts = text.split(separator)
                
                chunks = []
                current_chunk = ""
                
                for part in parts:
                    # If adding this part exceeds chunk_size
                    if len(current_chunk) + len(part) + len(separator) > self.chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                            
                            # Add overlap from end of current chunk
                            overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                            current_chunk = overlap_text + separator + part
                        else:
                            current_chunk = part
                    else:
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                
                if len(chunks) > 1:  # Successfully split
                    return chunks
        
        # Fallback: force split at chunk_size
        chunks = []
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks


# Test standalone
if __name__ == "__main__":
    sample_text = """
    Udatha Venkatasai
    Chennai, Tamil Nadu
    
    Professional Summary
    Early-career Java Backend Developer with 1+ years of Experience...
    
    Technical Skills
    Languages: Java, SQL, JavaScript
    Backend: Spring Boot, REST APIs
    """
    
    chunker = ResumeChunker(chunk_size=200, chunk_overlap=30)
    chunks = chunker.chunk_resume(sample_text, "test.pdf")
    
    for chunk in chunks:
        print(f"\n--- {chunk.chunk_id} ({chunk.metadata['section']}) ---")
        print(chunk.text)