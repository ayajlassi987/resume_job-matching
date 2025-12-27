# API Documentation

## Overview

The Interview Copilot API provides REST endpoints for resume matching, interview question generation, and answer evaluation.

Base URL: `http://localhost:5000/api`

## Endpoints

### Health Check

**GET** `/api/health`

Check the health status of the API and its components.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "components": {
    "rag_engine": "ok",
    "answer_evaluator": "ok",
    "embedding_engine": "ok",
    "vector_store": "ok"
  }
}
```

### System Statistics

**GET** `/api/stats`

Get system statistics and performance metrics.

**Response:**
```json
{
  "total_questions_generated": 150,
  "total_answers_evaluated": 75,
  "average_response_time": 1.23
}
```

### Resume Matching

**POST** `/api/match`

Match a resume against multiple job descriptions.

**Request Body:**
```json
{
  "resume": "Software engineer with 5 years of Python experience...",
  "job_descriptions": [
    "Looking for a Python developer...",
    "Senior software engineer position..."
  ],
  "top_k": 3
}
```

**Response:**
```json
{
  "matches": [
    {
      "job_index": 0,
      "job_description": "Looking for a Python developer...",
      "score": 0.95
    }
  ],
  "total_matched": 1
}
```

### Question Generation

**POST** `/api/generate-questions`

Generate tailored interview questions using RAG.

**Request Body:**
```json
{
  "resume": "Software engineer with ML experience...",
  "job_description": "Looking for a Python ML engineer...",
  "question_type": "technical",
  "top_k": 8,
  "use_hybrid_search": true,
  "use_reranking": true
}
```

**Parameters:**
- `question_type`: One of `"technical"`, `"behavioral"`, `"system_design"`, or `"default"`
- `top_k`: Number of retrieved questions to use (1-20)
- `use_hybrid_search`: Override hybrid search setting (optional)
- `use_reranking`: Override reranking setting (optional)

**Response:**
```json
{
  "questions": [
    "Describe your experience with Python machine learning frameworks.",
    "How would you design a scalable ML pipeline?",
    "..."
  ],
  "retrieved_questions": [
    "Tell me about a Python project...",
    "..."
  ],
  "question_type": "technical"
}
```

### Answer Evaluation

**POST** `/api/evaluate-answer`

Evaluate a candidate's answer to an interview question.

**Request Body:**
```json
{
  "answer": "I have 3 years of Python experience. I've built REST APIs using Flask...",
  "question": "Describe your Python experience.",
  "expected_keywords": ["Python", "Flask", "REST API"],
  "expected_skills": ["Python", "Flask"],
  "context": "Python developer position requiring Flask experience."
}
```

**Response:**
```json
{
  "overall_score": 0.85,
  "semantic_similarity": 0.82,
  "skill_coverage": 0.90,
  "keyword_match": 0.75,
  "coherence_score": 0.88,
  "completeness_score": 0.85,
  "extracted_skills": ["Python", "Flask"],
  "matched_keywords": ["Python", "Flask", "REST API"],
  "missing_keywords": [],
  "feedback": [
    "Excellent answer! The response demonstrates strong understanding.",
    "Good coverage of relevant skills."
  ],
  "suggestions": [
    "Consider adding more specific examples of your work."
  ]
}
```

## Error Responses

All endpoints may return error responses in the following format:

```json
{
  "error": "Error message description"
}
```

**Status Codes:**
- `400`: Bad Request (validation error)
- `404`: Not Found (endpoint doesn't exist)
- `500`: Internal Server Error

## Rate Limiting

The API supports rate limiting (configurable via `RATE_LIMIT_PER_MINUTE` environment variable). Default: 60 requests per minute.

## Authentication

Currently, the API does not require authentication. For production deployments, consider adding API key authentication.

## Example Usage

### Using curl

```bash
# Health check
curl http://localhost:5000/api/health

# Generate questions
curl -X POST http://localhost:5000/api/generate-questions \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Python developer...",
    "job_description": "Looking for Python engineer...",
    "question_type": "technical"
  }'
```

### Using Python

```python
import requests

# Generate questions
response = requests.post(
    "http://localhost:5000/api/generate-questions",
    json={
        "resume": "Python developer with ML experience...",
        "job_description": "Looking for Python ML engineer...",
        "question_type": "technical"
    }
)

questions = response.json()["questions"]
print(questions)
```

