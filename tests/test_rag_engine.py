"""Tests for RAG engine."""
import pytest
from src.rag_engine import RAGEngine
from src.embedding_engine import EmbeddingEngine


@pytest.fixture
def rag_engine():
    """Create RAG engine instance for testing."""
    return RAGEngine(index_dir="vector_store/faiss_index")


def test_rag_engine_initialization(rag_engine):
    """Test RAG engine initialization."""
    assert rag_engine is not None
    assert rag_engine.emb is not None


def test_query_expansion(rag_engine):
    """Test query expansion with skills."""
    resume = "Python developer with experience in machine learning and Docker"
    job = "Looking for a Python ML engineer"
    
    expanded = rag_engine._expand_query_with_skills(resume, job)
    assert len(expanded) > len(resume + job)
    assert "python" in expanded.lower()


def test_parse_questions(rag_engine):
    """Test question parsing."""
    text = """
    1. What is your experience with Python?
    2. Tell me about a project you worked on.
    3. How do you handle debugging?
    """
    questions = rag_engine._parse(text)
    assert len(questions) > 0
    assert all(len(q) > 5 for q in questions)


def test_prompt_templates(rag_engine):
    """Test prompt template loading."""
    assert "technical" in rag_engine.prompt_templates
    assert "behavioral" in rag_engine.prompt_templates
    assert "default" in rag_engine.prompt_templates

