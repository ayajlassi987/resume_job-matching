"""Tests for resume parser."""
import pytest
from src.resume_parser import parse_resume, _extract_sections, _extract_contact_info


def test_parse_resume_text():
    """Test parsing resume from text."""
    resume_text = """
    John Doe
    Software Engineer
    Email: john@example.com
    Phone: 123-456-7890
    
    Experience:
    - Software Engineer at Company X (2020-2023)
    - Developed Python applications
    
    Skills:
    Python, Docker, AWS
    """
    
    result = parse_resume(text=resume_text, extract_sections=True)
    
    assert "clean_text" in result
    assert "raw_text" in result
    assert len(result["clean_text"]) > 0


def test_extract_sections():
    """Test section extraction."""
    text = """
    Summary: Experienced developer
    
    Experience:
    Worked at Company X
    
    Skills:
    Python, Java
    """
    
    sections = _extract_sections(text)
    assert "experience" in sections or "skills" in sections


def test_extract_contact_info():
    """Test contact information extraction."""
    text = "Contact: john@example.com, Phone: 123-456-7890"
    contact = _extract_contact_info(text)
    
    assert "email" in contact or "phone" in contact

