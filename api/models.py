"""Pydantic models for API request/response validation."""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


class ResumeMatchRequest(BaseModel):
    """Request model for resume-job matching."""
    resume: str = Field(..., description="Resume text content")
    job_descriptions: List[str] = Field(..., description="List of job description texts")
    top_k: int = Field(3, ge=1, le=10, description="Number of top matches to return")
    
    @validator('resume')
    def validate_resume(cls, v):
        if not v or len(v.strip()) < 10:
            raise ValueError("Resume text must be at least 10 characters")
        return v.strip()
    
    @validator('job_descriptions')
    def validate_jobs(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one job description is required")
        return v


class ResumeMatchResponse(BaseModel):
    """Response model for resume-job matching."""
    matches: List[Dict[str, Any]] = Field(..., description="List of matched jobs with scores")
    total_matched: int = Field(..., description="Total number of matches")


class QuestionGenerationRequest(BaseModel):
    """Request model for question generation."""
    resume: str = Field(..., description="Resume text content")
    job_description: str = Field(..., description="Job description text")
    question_type: str = Field("default", description="Type of questions: technical, behavioral, system_design, default")
    top_k: int = Field(8, ge=1, le=20, description="Number of retrieved questions to use")
    use_hybrid_search: Optional[bool] = Field(None, description="Override hybrid search setting")
    use_reranking: Optional[bool] = Field(None, description="Override reranking setting")
    
    @validator('question_type')
    def validate_question_type(cls, v):
        allowed_types = ["technical", "behavioral", "system_design", "default"]
        if v not in allowed_types:
            raise ValueError(f"question_type must be one of: {', '.join(allowed_types)}")
        return v


class QuestionGenerationResponse(BaseModel):
    """Response model for question generation."""
    questions: List[str] = Field(..., description="Generated interview questions")
    retrieved_questions: List[str] = Field(default_factory=list, description="Retrieved questions used for RAG")
    question_type: str = Field(..., description="Type of questions generated")


class AnswerEvaluationRequest(BaseModel):
    """Request model for answer evaluation."""
    answer: str = Field(..., description="Candidate's answer text")
    question: str = Field(..., description="The interview question")
    expected_keywords: Optional[List[str]] = Field(None, description="Expected keywords in the answer")
    expected_skills: Optional[List[str]] = Field(None, description="Expected skills to be demonstrated")
    context: Optional[str] = Field(None, description="Additional context (resume, job description)")
    
    @validator('answer')
    def validate_answer(cls, v):
        if not v or len(v.strip()) < 5:
            raise ValueError("Answer text must be at least 5 characters")
        return v.strip()


class AnswerEvaluationResponse(BaseModel):
    """Response model for answer evaluation."""
    overall_score: float = Field(..., ge=0.0, le=1.0, description="Overall evaluation score (0-1)")
    semantic_similarity: float = Field(..., ge=0.0, le=1.0, description="Semantic similarity score")
    skill_coverage: float = Field(..., ge=0.0, le=1.0, description="Skill coverage score")
    keyword_match: float = Field(..., ge=0.0, le=1.0, description="Keyword match score")
    coherence_score: float = Field(..., ge=0.0, le=1.0, description="Coherence score")
    completeness_score: float = Field(..., ge=0.0, le=1.0, description="Completeness score")
    extracted_skills: List[str] = Field(default_factory=list, description="Skills extracted from answer")
    matched_keywords: List[str] = Field(default_factory=list, description="Matched keywords")
    missing_keywords: List[str] = Field(default_factory=list, description="Missing keywords")
    feedback: List[str] = Field(default_factory=list, description="Evaluation feedback")
    suggestions: List[str] = Field(default_factory=list, description="Improvement suggestions")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    components: Dict[str, str] = Field(..., description="Component statuses")


class StatsResponse(BaseModel):
    """System statistics response model."""
    total_questions_generated: int = Field(..., description="Total questions generated")
    total_answers_evaluated: int = Field(..., description="Total answers evaluated")
    average_response_time: float = Field(..., description="Average response time in seconds")

