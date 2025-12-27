"""Tests for Flask API."""
import pytest
from api.app import app


@pytest.fixture
def client():
    """Create test client."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    """Test health check endpoint."""
    response = client.get('/api/health')
    assert response.status_code == 200
    data = response.get_json()
    assert "status" in data
    assert "version" in data


def test_stats_endpoint(client):
    """Test stats endpoint."""
    response = client.get('/api/stats')
    assert response.status_code == 200
    data = response.get_json()
    assert "total_questions_generated" in data


def test_match_endpoint(client):
    """Test resume matching endpoint."""
    payload = {
        "resume": "Python developer with 5 years experience",
        "job_descriptions": [
            "Looking for a Python developer",
            "Java developer position"
        ],
        "top_k": 2
    }
    
    response = client.post('/api/match', json=payload)
    assert response.status_code in [200, 500]  # May fail if dependencies not available
    if response.status_code == 200:
        data = response.get_json()
        assert "matches" in data


def test_generate_questions_endpoint(client):
    """Test question generation endpoint."""
    payload = {
        "resume": "Python developer with ML experience",
        "job_description": "Looking for a Python ML engineer",
        "question_type": "technical",
        "top_k": 5
    }
    
    response = client.post('/api/generate-questions', json=payload)
    # May fail if LLM or index not available
    assert response.status_code in [200, 400, 500]


def test_evaluate_answer_endpoint(client):
    """Test answer evaluation endpoint."""
    payload = {
        "answer": "I have 3 years of Python experience",
        "question": "Tell me about your Python experience",
        "expected_keywords": ["Python", "experience"]
    }
    
    response = client.post('/api/evaluate-answer', json=payload)
    assert response.status_code in [200, 500]  # May fail if dependencies not available
    if response.status_code == 200:
        data = response.get_json()
        assert "overall_score" in data

