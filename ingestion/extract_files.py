import logging
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import pymupdf

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PDFExtractor:
    """Lightweight text extractor for PDF documents."""
    
    def extract_pdf_content(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract content from a single PDF file."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Extracting content from: {pdf_path.name}")
        start_time = time.time()
        page_texts = []

        with pymupdf.open(pdf_path) as document:
            page_count = len(document)
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text").strip()
                if page_text:
                    page_texts.append(f"## Page {page_number}\n\n{page_text}")

        end_time = time.time()
        content_text = "\n\n".join(page_texts)
        
        metadata = {
            "source": str(pdf_path),
            "title": pdf_path.stem,
            "processing_time": round(end_time - start_time, 2),
            "pages": page_count,
            "text_pages": len(page_texts),
            "characters": len(content_text),
            "extraction_method": "pymupdf",
            "content_type": "pdf"
        }
        return content_text, metadata
    

def create_pdf_extractor() -> PDFExtractor:
    """Create the lightweight PDF text extractor."""
    return PDFExtractor()
