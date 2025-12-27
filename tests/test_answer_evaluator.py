"""Tests for answer evaluator."""
import pytest
from src.answer_evaluator import AnswerEvaluator


@pytest.fixture
def evaluator():
    """Create answer evaluator instance."""
    return AnswerEvaluator()


def test_evaluate_answer(evaluator):
    """Test answer evaluation."""
    answer = "I have 5 years of experience with Python. I've worked on multiple projects using Django and Flask."
    question = "Tell me about your Python experience."
    
    result = evaluator.evaluate_answer(answer, question)
    
    assert "overall_score" in result
    assert 0.0 <= result["overall_score"] <= 1.0
    assert "feedback" in result
    assert isinstance(result["feedback"], list)


def test_calculate_coherence(evaluator):
    """Test coherence calculation."""
    answer = "This is a well-structured answer with multiple sentences. It explains the concept clearly."
    coherence = evaluator._calculate_coherence(answer)
    
    assert 0.0 <= coherence <= 1.0


def test_calculate_completeness(evaluator):
    """Test completeness calculation."""
    answer = "I have experience with Python and machine learning."
    question = "What is your experience with Python?"
    
    completeness = evaluator._calculate_completeness(answer, question)
    
    assert 0.0 <= completeness <= 1.0


def test_match_keywords(evaluator):
    """Test keyword matching."""
    answer = "I used Python and Docker for the project"
    keywords = ["Python", "Docker", "Kubernetes"]
    
    matched, missing = evaluator._match_keywords(answer, keywords)
    
    assert "Python" in matched or "python" in [k.lower() for k in matched]
    assert isinstance(missing, list)

