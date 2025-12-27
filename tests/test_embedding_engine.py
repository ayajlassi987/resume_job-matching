"""Tests for embedding engine."""
import pytest
import numpy as np
from src.embedding_engine import EmbeddingEngine


@pytest.fixture
def embedding_engine():
    """Create embedding engine instance for testing."""
    return EmbeddingEngine(use_openai=False, index_dir="vector_store/faiss_index")


def test_embedding_engine_initialization(embedding_engine):
    """Test embedding engine initialization."""
    assert embedding_engine is not None
    assert embedding_engine.use_openai is False


def test_encode_texts(embedding_engine):
    """Test text encoding."""
    texts = ["Hello world", "Test embedding"]
    embeddings = embedding_engine.encode(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] > 0


def test_build_index(embedding_engine, tmp_path):
    """Test index building."""
    docs = ["Document 1", "Document 2", "Document 3"]
    embedding_engine.index_dir = tmp_path / "test_index"
    embedding_engine.build_index(docs)
    
    assert len(embedding_engine._docs) == len(docs)
    assert embedding_engine._index is not None


def test_search(embedding_engine):
    """Test search functionality."""
    # Build a small test index
    docs = ["Python programming", "Machine learning", "Data science"]
    embedding_engine.build_index(docs)
    
    results = embedding_engine.search("Python", top_k=2)
    assert len(results) > 0
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)

