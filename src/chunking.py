"""Intelligent chunking utilities for long documents.

This module provides semantic chunking and context window management
for resumes and job descriptions.
"""
from typing import List
import re


def semantic_chunk_text(text: str, max_chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into semantic chunks with overlap.
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    # Try to split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # Add overlap between chunks
    if len(chunks) > 1 and overlap > 0:
        overlapped_chunks = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_end = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            overlapped_chunks.append(prev_end + " " + chunks[i])
        return overlapped_chunks
    
    return chunks


def chunk_resume(resume_text: str, max_chunk_size: int = 1000) -> List[str]:
    """
    Chunk resume text by sections (Education, Experience, Skills, etc.).
    
    Args:
        resume_text: Full resume text
        max_chunk_size: Maximum characters per chunk
    
    Returns:
        List of resume section chunks
    """
    # Common resume section headers
    section_patterns = [
        r'(?i)^(?:education|academic|qualifications)',
        r'(?i)^(?:experience|work\s+history|employment)',
        r'(?i)^(?:skills|technical\s+skills|competencies)',
        r'(?i)^(?:projects|portfolio)',
        r'(?i)^(?:certifications|certificates)',
        r'(?i)^(?:summary|objective|profile)',
    ]
    
    lines = resume_text.split('\n')
    sections = []
    current_section = []
    current_header = None
    
    for line in lines:
        is_header = any(re.match(pattern, line.strip()) for pattern in section_patterns)
        
        if is_header:
            if current_section:
                sections.append((current_header, '\n'.join(current_section)))
            current_header = line.strip()
            current_section = []
        else:
            current_section.append(line)
    
    if current_section:
        sections.append((current_header or "Other", '\n'.join(current_section)))
    
    # If no sections found, use semantic chunking
    if not sections:
        return semantic_chunk_text(resume_text, max_chunk_size)
    
    # Return section texts
    return [section_text for _, section_text in sections if section_text.strip()]


def manage_context_window(texts: List[str], max_tokens: int = 3000, chars_per_token: int = 4) -> List[str]:
    """
    Manage context window by truncating or prioritizing texts.
    
    Args:
        texts: List of texts to manage
        max_tokens: Maximum tokens allowed
        chars_per_token: Average characters per token
    
    Returns:
        List of texts that fit within context window
    """
    max_chars = max_tokens * chars_per_token
    selected_texts = []
    current_length = 0
    
    for text in texts:
        if current_length + len(text) <= max_chars:
            selected_texts.append(text)
            current_length += len(text)
        else:
            # Truncate the last text to fit
            remaining = max_chars - current_length
            if remaining > 100:  # Only add if meaningful space remains
                selected_texts.append(text[:remaining] + "...")
            break
    
    return selected_texts

