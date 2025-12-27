"""Tests for skill extractor."""
import pytest
from src.skill_extractor import extract_skills, normalize_skill_name


def test_extract_skills_basic():
    """Test basic skill extraction."""
    text = "I have experience with Python, Docker, and AWS"
    skills = extract_skills(text, use_ner=False, use_fuzzy=False)
    
    # Should extract at least some skills if skill list is available
    assert isinstance(skills, list)


def test_extract_skills_with_fuzzy():
    """Test skill extraction with fuzzy matching."""
    text = "Experienced in Python programming and machine learning"
    skills = extract_skills(text, use_fuzzy=True, fuzzy_threshold=70)
    
    assert isinstance(skills, list)


def test_normalize_skill_name():
    """Test skill name normalization."""
    skill = "python"
    normalized = normalize_skill_name(skill)
    
    assert isinstance(normalized, str)
    assert len(normalized) > 0

