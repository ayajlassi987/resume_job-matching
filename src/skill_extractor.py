"""Enhanced skill extraction with NER, fuzzy matching, and taxonomy support."""
import json
import re
from typing import List, Set, Dict, Tuple
from pathlib import Path

try:
    from thefuzz import fuzz, process
except ImportError:
    fuzz = None
    process = None

try:
    import spacy
    nlp = None
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        pass
except ImportError:
    nlp = None

# Load skill list
SKILLS = set()
SKILL_TAXONOMY = {}

try:
    skill_file = Path("data/skills/skill_list.json")
    if skill_file.exists():
        with open(skill_file, "r", encoding="utf-8") as f:
            skills_data = json.load(f)
            if isinstance(skills_data, list):
                SKILLS = set(skills_data)
            elif isinstance(skills_data, dict):
                SKILLS = set(skills_data.get("skills", []))
                SKILL_TAXONOMY = skills_data.get("taxonomy", {})
except Exception:
    pass


def extract_skills(
    text: str,
    use_ner: bool = False,  # Disabled by default to avoid hallucinations
    use_fuzzy: bool = False,  # Disabled by default to avoid hallucinations
    fuzzy_threshold: int = 90,  # Higher threshold for stricter matching
    include_proficiency: bool = False
) -> List[str]:
    """
    Extract skills from text using strict matching against known skill list.
    
    Args:
        text: Input text to extract skills from
        use_ner: Use Named Entity Recognition (disabled by default)
        use_fuzzy: Use fuzzy matching for skill variations (disabled by default)
        fuzzy_threshold: Similarity threshold for fuzzy matching (0-100, default 90)
        include_proficiency: Extract proficiency levels
    
    Returns:
        List of extracted skills (only from known skill list)
    """
    if not SKILLS:
        return []  # Return empty if no skill list available
    
    found_skills = set()
    text_lower = text.lower()
    
    # Method 1: Exact keyword matching (primary method - most reliable)
    for skill in SKILLS:
        skill_lower = skill.lower()
        # Check for word boundary matches only
        if re.search(r'\b' + re.escape(skill_lower) + r'\b', text_lower):
            found_skills.add(skill)
    
    # Method 2: NER-based extraction (only if enabled and skill list exists)
    if use_ner and nlp and SKILLS:
        ner_skills = _extract_skills_ner(text)
        # Only add skills that are in our known skill list
        found_skills.update(ner_skills & SKILLS)
    
    # Method 3: Fuzzy matching (only if enabled, with high threshold)
    if use_fuzzy and fuzz and process and SKILLS:
        fuzzy_skills = _extract_skills_fuzzy(text, fuzzy_threshold)
        # Only add skills that match known skills
        found_skills.update(fuzzy_skills & SKILLS)
    
    # Extract proficiency levels if requested
    if include_proficiency:
        return _add_proficiency_levels(list(found_skills), text)
    
    return list(found_skills)


def _extract_skills_ner(text: str) -> Set[str]:
    """Extract skills using Named Entity Recognition - only returns known skills."""
    if not nlp or not SKILLS:
        return set()
    
    doc = nlp(text)
    skills = set()
    
    # Look for noun phrases that match known skills exactly
    for chunk in doc.noun_chunks:
        chunk_text = chunk.text.strip()
        chunk_lower = chunk_text.lower()
        # Only add if it's an exact match or substring of a known skill
        for skill in SKILLS:
            skill_lower = skill.lower()
            # Strict matching: skill must be in chunk or chunk must be in skill
            if skill_lower == chunk_lower or (len(chunk_lower) > 3 and skill_lower in chunk_lower):
                skills.add(skill)
    
    # Look for specific patterns but only match known skills
    patterns = [
        r'(?:proficient|experienced|skilled|expert)\s+(?:in|with|at)\s+([A-Za-z\s]{2,30})',
        r'(?:knowledge|experience)\s+(?:of|in|with)\s+([A-Za-z\s]{2,30})',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            match_clean = match.strip()
            # Only check against known skills - strict matching
            for skill in SKILLS:
                skill_lower = skill.lower()
                match_lower = match_clean.lower()
                # Only match if skill is mentioned in the match or vice versa
                if skill_lower in match_lower or match_lower in skill_lower:
                    # Additional validation: match should be reasonable length
                    if 2 <= len(match_clean) <= 30:
                        skills.add(skill)
    
    return skills


def _extract_skills_fuzzy(text: str, threshold: int = 90) -> Set[str]:
    """Extract skills using fuzzy matching - strict matching only."""
    if not process or not SKILLS:
        return set()
    
    # Extract potential skill mentions (capitalized words, technical terms)
    # Be more selective - look for technical terms
    words = re.findall(r'\b[A-Z][a-z]{3,}\b', text)  # Capitalized words (min 4 chars)
    words.extend(re.findall(r'\b[a-z]{3,}\.[a-z]+\b', text.lower()))  # Abbreviations
    
    found_skills = set()
    
    for word in words:
        if len(word) < 4:  # Minimum length to avoid short words
            continue
        
        # Find best matching skill with high threshold
        matches = process.extract(word, SKILLS, limit=1, scorer=fuzz.ratio)
        if matches and matches[0][1] >= threshold:
            # Additional validation: the match should be reasonable
            matched_skill = matches[0][0]
            if len(matched_skill) >= 3:  # Skill name should be at least 3 chars
                found_skills.add(matched_skill)
    
    return found_skills


def _add_proficiency_levels(skills: List[str], text: str) -> List[Dict]:
    """Add proficiency levels to extracted skills."""
    proficiency_keywords = {
        "expert": ["expert", "advanced", "senior", "master", "proficient"],
        "intermediate": ["intermediate", "experienced", "familiar", "comfortable"],
        "beginner": ["beginner", "basic", "learning", "novice"]
    }
    
    result = []
    text_lower = text.lower()
    
    for skill in skills:
        skill_info = {"skill": skill, "proficiency": "unknown"}
        
        # Check for proficiency indicators near the skill mention
        skill_pos = text_lower.find(skill.lower())
        if skill_pos != -1:
            # Look in surrounding context (50 chars before and after)
            context_start = max(0, skill_pos - 50)
            context_end = min(len(text_lower), skill_pos + len(skill) + 50)
            context = text_lower[context_start:context_end]
            
            for level, keywords in proficiency_keywords.items():
                if any(keyword in context for keyword in keywords):
                    skill_info["proficiency"] = level
                    break
        
        result.append(skill_info)
    
    return result


def get_skill_taxonomy(skill: str) -> Dict:
    """Get taxonomy information for a skill."""
    if not SKILL_TAXONOMY:
        return {}
    
    skill_lower = skill.lower()
    for category, skills_list in SKILL_TAXONOMY.items():
        if skill_lower in [s.lower() for s in skills_list]:
            return {"category": category, "related_skills": skills_list}
    
    return {}


def normalize_skill_name(skill: str) -> str:
    """Normalize skill name to canonical form."""
    # Find best match in skill list
    if not SKILLS:
        return skill
    
    skill_lower = skill.lower()
    
    # Exact match
    for s in SKILLS:
        if s.lower() == skill_lower:
            return s
    
    # Fuzzy match
    if process:
        matches = process.extract(skill, SKILLS, limit=1, scorer=fuzz.ratio)
        if matches and matches[0][1] >= 90:
            return matches[0][0]
    
    return skill
