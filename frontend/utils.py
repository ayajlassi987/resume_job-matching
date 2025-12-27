"""Utility functions for the Streamlit frontend."""
from typing import Optional
import io

try:
    from pdfminer.high_level import extract_text as extract_pdf_text
except ImportError:
    extract_pdf_text = None

try:
    from docx import Document
except ImportError:
    Document = None


def extract_text_from_file(uploaded_file) -> Optional[str]:
    """
    Extract text from uploaded file (PDF, DOCX, or TXT).
    
    Args:
        uploaded_file: Streamlit uploaded file object
    
    Returns:
        Extracted text or None if error
    """
    if uploaded_file is None:
        return None
    
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            if extract_pdf_text is None:
                return None
            # Read PDF
            file_bytes = uploaded_file.read()
            return extract_pdf_text(io.BytesIO(file_bytes))
        
        elif file_extension in ['docx', 'doc']:
            if Document is None:
                return None
            # Read DOCX
            file_bytes = uploaded_file.read()
            doc = Document(io.BytesIO(file_bytes))
            return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        elif file_extension == 'txt':
            # Read text file
            return uploaded_file.read().decode('utf-8', errors='ignore')
        
        else:
            return None
    
    except Exception as e:
        return None


def format_score(score: float) -> str:
    """Format score as percentage."""
    return f"{score * 100:.1f}%"


def format_response_time(seconds: float) -> str:
    """Format response time in human-readable format."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    else:
        return f"{seconds:.2f}s"

