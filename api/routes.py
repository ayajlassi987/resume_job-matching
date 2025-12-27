"""API routes for the Flask application."""
import time
from flask import Blueprint, request, jsonify
from typing import Dict, Any

from api.models import (
    ResumeMatchRequest, ResumeMatchResponse,
    QuestionGenerationRequest, QuestionGenerationResponse,
    AnswerEvaluationRequest, AnswerEvaluationResponse,
    HealthResponse, StatsResponse
)
from src.matcher import match_pipeline
from src.rag_engine import RAGEngine
from src.answer_evaluator import AnswerEvaluator
from src.embedding_engine import EmbeddingEngine
from config import RAG_TOP_K, ENABLE_HYBRID_SEARCH, ENABLE_RERANKING

# Initialize components (use local embeddings only, no OpenAI)
rag_engine = RAGEngine()
rag_engine.emb.use_openai = False  # Force local embeddings
answer_evaluator = AnswerEvaluator()

# Statistics tracking
stats = {
    "total_questions_generated": 0,
    "total_answers_evaluated": 0,
    "total_requests": 0,
    "response_times": []
}

# Create blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')


@api_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    components = {
        "rag_engine": "ok",
        "answer_evaluator": "ok",
        "embedding_engine": "ok"
    }
    
    # Test embedding engine
    try:
        emb = EmbeddingEngine()
        emb.load_index()
        components["vector_store"] = "ok"
    except Exception as e:
        components["vector_store"] = f"error: {str(e)}"
    
    response = HealthResponse(
        status="healthy",
        version="1.0.0",
        components=components
    )
    return jsonify(response.dict()), 200


@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics."""
    avg_response_time = (
        sum(stats["response_times"]) / len(stats["response_times"])
        if stats["response_times"] else 0.0
    )
    
    response = StatsResponse(
        total_questions_generated=stats["total_questions_generated"],
        total_answers_evaluated=stats["total_answers_evaluated"],
        average_response_time=avg_response_time
    )
    return jsonify(response.dict()), 200


@api_bp.route('/match', methods=['POST'])
def match_resume():
    """Match resume to job descriptions."""
    start_time = time.time()
    stats["total_requests"] += 1
    
    try:
        # Validate request
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "Request body is required"}), 400
        
        match_request = ResumeMatchRequest(**req_data)
        
        # Perform matching
        matches = match_pipeline(
            [match_request.resume],
            match_request.job_descriptions
        )
        
        # Format response - filter by score >= 0.7 and return requested number
        match_results = []
        if matches and len(matches) > 0:
            # matches[0] is already filtered to scores >= 0.7 and sorted by score
            top_matches = matches[0][:match_request.top_k]  # Get top_k matches
            for idx, score in top_matches:
                # Ensure score is between 0.7 and 1.0
                if 0.7 <= score <= 1.0:
                    match_results.append({
                        "job_index": int(idx),
                        "job_description": match_request.job_descriptions[idx][:200] + "...",
                        "score": round(score, 3)  # Round to 3 decimal places
                    })
        
        response = ResumeMatchResponse(
            matches=match_results,
            total_matched=len(match_results)
        )
        
        stats["response_times"].append(time.time() - start_time)
        if len(stats["response_times"]) > 100:
            stats["response_times"] = stats["response_times"][-100:]
        
        return jsonify(response.dict()), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route('/generate-questions', methods=['POST'])
def generate_questions():
    """Generate interview questions using RAG."""
    start_time = time.time()
    stats["total_requests"] += 1
    
    try:
        # Validate request
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "Request body is required"}), 400
        
        gen_request = QuestionGenerationRequest(**req_data)
        
        # Generate questions
        questions = rag_engine.generate_questions(
            resume=gen_request.resume,
            job=gen_request.job_description,
            top_k=gen_request.top_k,
            question_type=gen_request.question_type,
            use_hybrid_search=gen_request.use_hybrid_search,
            use_reranking=gen_request.use_reranking
        )
        
        # Get retrieved questions for reference
        retrieved = rag_engine.emb.search(
            rag_engine._expand_query_with_skills(gen_request.resume, gen_request.job_description),
            top_k=gen_request.top_k
        )
        retrieved_questions = [q for q, _ in retrieved]
        
        response = QuestionGenerationResponse(
            questions=questions,
            retrieved_questions=retrieved_questions[:5],  # Limit to top 5
            question_type=gen_request.question_type
        )
        
        stats["total_questions_generated"] += len(questions)
        stats["response_times"].append(time.time() - start_time)
        if len(stats["response_times"]) > 100:
            stats["response_times"] = stats["response_times"][-100:]
        
        return jsonify(response.dict()), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


@api_bp.route('/evaluate-answer', methods=['POST'])
def evaluate_answer():
    """Evaluate a candidate's answer."""
    start_time = time.time()
    stats["total_requests"] += 1
    
    try:
        # Validate request
        req_data = request.get_json()
        if not req_data:
            return jsonify({"error": "Request body is required"}), 400
        
        eval_request = AnswerEvaluationRequest(**req_data)
        
        # Evaluate answer
        evaluation = answer_evaluator.evaluate_answer(
            candidate_answer=eval_request.answer,
            question=eval_request.question,
            expected_keywords=eval_request.expected_keywords,
            expected_skills=eval_request.expected_skills,
            context=eval_request.context
        )
        
        response = AnswerEvaluationResponse(**evaluation)
        
        stats["total_answers_evaluated"] += 1
        stats["response_times"].append(time.time() - start_time)
        if len(stats["response_times"]) > 100:
            stats["response_times"] = stats["response_times"][-100:]
        
        return jsonify(response.dict()), 200
    
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

