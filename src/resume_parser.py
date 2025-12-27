"""Enhanced resume parser supporting multiple formats and structured extraction."""
import re
from typing import Dict, List, Optional
from pathlib import Path

try:
    from pdfminer.high_level import extract_text as extract_pdf_text
except ImportError:
    extract_pdf_text = None

try:
    from docx import Document
except ImportError:
    Document = None

from modules.utils import clean_text


def parse_resume(
    text: Optional[str] = None,
    file_path: Optional[str] = None,
    extract_sections: bool = True
) -> Dict:
    """
    Parse resume from text or file path.
    
    Args:
        text: Resume text content
        file_path: Path to resume file (PDF, DOCX, or TXT)
        extract_sections: Whether to extract structured sections
    
    Returns:
        Dictionary with parsed resume data
    """
    # Load text from file if provided
    if file_path and not text:
        text = _load_resume_file(file_path)
    
    if not text:
        raise ValueError("Either text or file_path must be provided")
    
    result = {
        "clean_text": clean_text(text),
        "raw_text": text
    }
    
    if extract_sections:
        result.update(_extract_sections(text))
    
    return result


def _load_resume_file(file_path: str) -> str:
    """Load resume text from file (PDF, DOCX, or TXT)."""
    path = Path(file_path)
    
    if path.suffix.lower() == '.pdf':
        if extract_pdf_text is None:
            raise ImportError("pdfminer.six not installed. Install with: pip install pdfminer.six")
        return extract_pdf_text(file_path)
    
    elif path.suffix.lower() in ['.docx', '.doc']:
        if Document is None:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        doc = Document(file_path)
        return '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    
    elif path.suffix.lower() == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")


def _extract_sections(text: str) -> Dict:
    """Extract structured sections from resume text."""
    sections = {
        "summary": "",
        "experience": [],
        "education": [],
        "skills": [],
        "projects": [],
        "certifications": [],
        "contact": {}
    }
    
    lines = text.split('\n')
    current_section = None
    current_content = []
    
    # Common section headers
    section_patterns = {
        "summary": r'(?i)^(?:summary|objective|profile|about)',
        "experience": r'(?i)^(?:experience|work\s+history|employment|professional\s+experience)',
        "education": r'(?i)^(?:education|academic|qualifications|academic\s+background)',
        "skills": r'(?i)^(?:skills|technical\s+skills|competencies|proficiencies)',
        "projects": r'(?i)^(?:projects|portfolio|key\s+projects)',
        "certifications": r'(?i)^(?:certifications|certificates|licenses)',
        "contact": r'(?i)^(?:contact|personal\s+information)'
    }
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this line is a section header
        found_section = None
        for section_name, pattern in section_patterns.items():
            if re.match(pattern, line):
                found_section = section_name
                break
        
        if found_section:
            # Save previous section content
            if current_section and current_content:
                sections[current_section] = _process_section_content(current_section, current_content)
            current_section = found_section
            current_content = []
        elif current_section:
            current_content.append(line)
        else:
            # Content before any section header goes to summary
            if not sections["summary"]:
                sections["summary"] = line
            else:
                sections["summary"] += " " + line
    
    # Save last section
    if current_section and current_content:
        sections[current_section] = _process_section_content(current_section, current_content)
    
    # Extract contact information
    sections["contact"] = _extract_contact_info(text)
    
    # Normalize dates and locations
    sections = _normalize_dates_locations(sections)
    
    return sections


def _process_section_content(section_name: str, content: List[str]) -> any:
    """Process section content based on section type."""
    text = '\n'.join(content)
    
    if section_name == "experience":
        return _parse_experience(text)
    elif section_name == "education":
        return _parse_education(text)
    elif section_name == "skills":
        return _parse_skills(text)
    elif section_name == "projects":
        return _parse_projects(text)
    elif section_name == "certifications":
        return _parse_certifications(text)
    else:
        return text


def _parse_experience(text: str) -> List[Dict]:
    """Parse experience section into structured entries."""
    entries = []
    # Simple parsing: split by common patterns
    # This is a basic implementation; can be enhanced with ML models
    parts = re.split(r'\n(?=[A-Z][a-z]+.*\d{4})', text)  # Split by potential job entries
    
    for part in parts:
        if len(part.strip()) < 20:
            continue
        
        # Try to extract company, title, dates
        lines = part.split('\n')
        if len(lines) >= 2:
            entry = {
                "title": lines[0].strip(),
                "company": lines[1].strip() if len(lines) > 1 else "",
                "description": '\n'.join(lines[2:]).strip(),
                "dates": _extract_dates(part)
            }
            entries.append(entry)
    
    return entries if entries else [{"description": text}]


def _parse_education(text: str) -> List[Dict]:
    """Parse education section."""
    entries = []
    parts = re.split(r'\n(?=[A-Z])', text)
    
    for part in parts:
        if len(part.strip()) < 10:
            continue
        
        entry = {
            "institution": part.split('\n')[0].strip(),
            "degree": "",
            "dates": _extract_dates(part),
            "description": part
        }
        entries.append(entry)
    
    return entries if entries else [{"description": text}]


def _parse_skills(text: str) -> List[str]:
    """Parse skills section into list."""
    # Split by common delimiters
    skills = re.split(r'[,;•\n]', text)
    return [s.strip() for s in skills if s.strip() and len(s.strip()) > 1]


def _parse_projects(text: str) -> List[Dict]:
    """Parse projects section."""
    projects = []
    parts = re.split(r'\n(?=[A-Z])', text)
    
    for part in parts:
        if len(part.strip()) < 10:
            continue
        projects.append({"name": part.split('\n')[0].strip(), "description": part})
    
    return projects if projects else [{"description": text}]


def _parse_certifications(text: str) -> List[str]:
    """Parse certifications section."""
    certs = re.split(r'\n', text)
    return [c.strip() for c in certs if c.strip()]


def _extract_contact_info(text: str) -> Dict:
    """Extract contact information."""
    contact = {}
    
    # Email
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    if emails:
        contact["email"] = emails[0]
    
    # Phone
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    if phones:
        contact["phone"] = phones[0] if isinstance(phones[0], str) else ''.join(phones[0])
    
    # LinkedIn
    linkedin_pattern = r'linkedin\.com/in/[\w-]+'
    linkedin = re.findall(linkedin_pattern, text)
    if linkedin:
        contact["linkedin"] = linkedin[0]
    
    return contact


def _extract_dates(text: str) -> str:
    """Extract date ranges from text."""
    # Pattern for dates like "Jan 2020 - Dec 2022" or "2020-2022"
    date_pattern = r'(\d{4}|\w+\s+\d{4})(?:\s*[-–—]\s*)?(\d{4}|\w+\s+\d{4}|present|current)?'
    dates = re.findall(date_pattern, text, re.IGNORECASE)
    if dates:
        return ' - '.join([d for d in dates[0] if d])
    return ""


def _normalize_dates_locations(sections: Dict) -> Dict:
    """Normalize dates and locations in sections."""
    # This is a placeholder for more sophisticated normalization
    # Can be enhanced with date parsing libraries
    return sections
