"""
PDF_EXTRACTOR.PY - Extract text from resume PDF

WHY pdfplumber over PyPDF2?
- Better text extraction quality
- Handles tables, columns better
- Preserves layout information
- More reliable with different PDF formats

WHAT HAPPENS HERE:
1. Open PDF file
2. Extract text from each page
3. Clean the text (remove extra whitespace, etc.)
4. Return clean text
"""

import pdfplumber
import re
from typing import Optional


class PDFExtractor:
    """Extracts and cleans text from PDF files"""
    
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract all text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Cleaned text string
            
        Flow:
            PDF File → pdfplumber opens it → Extract each page → 
            Combine pages → Clean text → Return
        """
        full_text = ""
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
                        print(f"  ✅ Extracted page {page_num + 1}: {len(page_text)} chars")
                    else:
                        print(f"  ⚠️ Page {page_num + 1}: No text found")
            
            # Clean the extracted text
            cleaned_text = self._clean_text(full_text)
            print(f"  📄 Total extracted: {len(cleaned_text)} characters")
            
            return cleaned_text
            
        except Exception as e:
            raise Exception(f"Failed to extract PDF: {str(e)}")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text.
        
        WHY CLEAN?
        - PDFs often have weird spacing
        - Multiple consecutive newlines
        - Special characters from PDF encoding
        - Extra whitespace from column layouts
        """
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace from each line
        lines = [line.strip() for line in text.split('\n')]
        text = '\n'.join(lines)
        
        # Remove any null characters
        text = text.replace('\x00', '')
        
        return text.strip()


# Test it standalone
if __name__ == "__main__":
    extractor = PDFExtractor()
    text = extractor.extract_text("data/resume.pdf")
    print("\n--- EXTRACTED TEXT ---")
    print(text[:1000])