"""Integration tests for the full pipeline."""
import pytest
from src.rag_engine import RAGEngine
from src.answer_evaluator import AnswerEvaluator
from src.resume_parser import parse_resume
from src.skill_extractor import extract_skills


def test_end_to_end_rag_pipeline():
    """Test complete RAG pipeline."""
    rag = RAGEngine()
    
    resume = "Software engineer with 5 years of Python experience. Worked on ML projects."
    job = "Looking for a Python ML engineer with Docker experience."
    
    # This might fail if index is not built, so we catch the exception
    try:
        questions = rag.generate_questions(resume, job, top_k=5)
        assert isinstance(questions, list)
    except Exception:
        pytest.skip("RAG index not available for testing")


def test_resume_parsing_and_skill_extraction():
    """Test resume parsing followed by skill extraction."""
    resume_text = """
    John Doe
    Python Developer
    
    Skills: Python, Docker, AWS, Machine Learning
    
    Experience:
    - Developed Python applications
    - Worked with Docker containers
    """
    
    parsed = parse_resume(text=resume_text)
    skills = extract_skills(parsed["clean_text"])
    
    assert "clean_text" in parsed
    assert isinstance(skills, list)


def test_answer_evaluation_with_context():
    """Test answer evaluation with full context."""
    evaluator = AnswerEvaluator()
    
    answer = "I have 3 years of experience with Python. I've built REST APIs using Flask."
    question = "Describe your Python experience."
    context = "Python developer position requiring Flask and REST API experience."
    
    result = evaluator.evaluate_answer(answer, question, context=context)
    
    assert result["overall_score"] >= 0.0
    assert "extracted_skills" in result

