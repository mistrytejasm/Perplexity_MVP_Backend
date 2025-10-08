import fitz
import io
from models.schemas import ExtractedContent
import logging
from typing import List

logger = logging.getLogger(__name__)

class DocumentContentExtractor:
    def __init__(self):
        self.min_text_threshold = 10

    def extract_from_pdf(self, file_path: str) -> ExtractedContent:
        """Extract Text from PDF file"""

        logger.info(f"Extracting Content from: {file_path}")

        try:
            doc = fitz.open(file_path)
            total_pages = len(doc)
            text_content = ""

            # Try direct text Extraction first
            for page_num in range(total_pages):
                page = doc[page_num]
                page_text = page.get_text()
                text_content += f"\n--- Page {page_num + 1} ---\n{page_text}"

            doc.close()

            # check if we get minigfull text
            clean_text = text_content.strip()
            if len(clean_text) > self.min_text_threshold:
                logger.info(f"Direct Text Extraction Successfull: {len(clean_text)} characters")

                return ExtractedContent(
                    text=clean_text,
                    total_pages=total_pages,
                    extraction_method="direct"
                )
            
            else:
                logger.warning(f"Direct extraction yielded only {len(clean_text)} characters.")
                return {
                    'text': clean_text,  # Return whatever was extracted
                    'warning': 'Limited text extracted. PDF may be image-based or poorly formatted.',
                    'pages_processed': total_pages
                }
            
        except Exception as e:
            logger.error(f"PDF Extraction Failed: {e}")
            raise Exception(f"Failed to extract content from PDF: {str(e)}")
        
    
