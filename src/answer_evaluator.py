"""Answer evaluation module for interview responses.

This module provides semantic similarity scoring, skill extraction,
keyword matching, and feedback generation for candidate answers.
"""
from typing import List, Dict, Tuple, Optional
import numpy as np
from src.embedding_engine import EmbeddingEngine
from src.skill_extractor import extract_skills


class AnswerEvaluator:
    """Evaluates candidate answers against expected criteria."""
    
    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None):
        """
        Initialize answer evaluator.
        
        Args:
            embedding_engine: Optional embedding engine for semantic similarity
        """
        # Use local embeddings only - disable OpenAI
        self.emb = embedding_engine or EmbeddingEngine(use_openai=False)
        self._technical_keywords_cache = {}
    
    def evaluate_answer(
        self,
        candidate_answer: str,
        question: str,
        expected_keywords: Optional[List[str]] = None,
        expected_skills: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Evaluate a candidate's answer comprehensively.
        
        Args:
            candidate_answer: The candidate's answer text
            question: The interview question asked
            expected_keywords: Expected technical keywords in the answer
            expected_skills: Expected skills to be demonstrated
            context: Additional context (resume, job description, etc.)
        
        Returns:
            Dictionary with evaluation scores and feedback
        """
        results = {
            "semantic_similarity": 0.0,
            "skill_coverage": 0.0,
            "keyword_match": 0.0,
            "coherence_score": 0.0,
            "completeness_score": 0.0,
            "overall_score": 0.0,
            "extracted_skills": [],
            "matched_keywords": [],
            "missing_keywords": [],
            "feedback": [],
            "suggestions": []
        }
        
        # Extract skills from answer - use strict matching only (no NER, no fuzzy)
        # This prevents hallucination of random words
        answer_skills = extract_skills(
            candidate_answer,
            use_ner=False,  # Disable NER to avoid hallucinations
            use_fuzzy=False  # Disable fuzzy matching to avoid hallucinations
        )
        results["extracted_skills"] = answer_skills
        
        # Semantic similarity scoring
        if context:
            results["semantic_similarity"] = self._calculate_semantic_similarity(
                candidate_answer, context
            )
        
        # Skill-based evaluation
        if expected_skills:
            results["skill_coverage"] = self._calculate_skill_coverage(
                answer_skills, expected_skills
            )
        
        # Keyword matching
        if expected_keywords:
            matched, missing = self._match_keywords(candidate_answer, expected_keywords)
            results["matched_keywords"] = matched
            results["missing_keywords"] = missing
            results["keyword_match"] = len(matched) / len(expected_keywords) if expected_keywords else 0.0
        
        # Coherence scoring
        results["coherence_score"] = self._calculate_coherence(candidate_answer)
        
        # Completeness scoring
        results["completeness_score"] = self._calculate_completeness(
            candidate_answer, question
        )
        
        # Calculate overall score (weighted average)
        weights = {
            "semantic_similarity": 0.2,
            "skill_coverage": 0.3,
            "keyword_match": 0.2,
            "coherence_score": 0.15,
            "completeness_score": 0.15
        }
        
        results["overall_score"] = (
            weights["semantic_similarity"] * results["semantic_similarity"] +
            weights["skill_coverage"] * results["skill_coverage"] +
            weights["keyword_match"] * results["keyword_match"] +
            weights["coherence_score"] * results["coherence_score"] +
            weights["completeness_score"] * results["completeness_score"]
        )
        
        # Generate feedback
        results["feedback"] = self._generate_feedback(results)
        results["suggestions"] = self._generate_suggestions(results, candidate_answer)
        
        return results
    
    def _calculate_semantic_similarity(self, answer: str, context: str) -> float:
        """Calculate semantic similarity between answer and context."""
        try:
            answer_emb = self.emb.encode([answer])
            context_emb = self.emb.encode([context])
            
            # Normalize embeddings
            answer_norm = answer_emb / (np.linalg.norm(answer_emb) + 1e-8)
            context_norm = context_emb / (np.linalg.norm(context_emb) + 1e-8)
            
            # Cosine similarity
            similarity = float(np.dot(answer_norm[0], context_norm[0]))
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]
        except Exception:
            return 0.0
    
    def _calculate_skill_coverage(self, answer_skills: List[str], expected_skills: List[str]) -> float:
        """Calculate how many expected skills are covered in the answer."""
        if not expected_skills:
            return 1.0
        
        answer_skills_lower = {s.lower() for s in answer_skills}
        expected_skills_lower = {s.lower() for s in expected_skills}
        
        matched = len(answer_skills_lower & expected_skills_lower)
        return matched / len(expected_skills_lower) if expected_skills_lower else 0.0
    
    def _match_keywords(self, answer: str, keywords: List[str]) -> Tuple[List[str], List[str]]:
        """Match keywords in answer (case-insensitive, partial matching)."""
        answer_lower = answer.lower()
        matched = []
        missing = []
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            # Check for exact match or word boundary match
            if keyword_lower in answer_lower:
                matched.append(keyword)
            else:
                # Check for partial match (keyword is part of a word)
                words = answer_lower.split()
                if any(keyword_lower in word or word in keyword_lower for word in words):
                    matched.append(keyword)
                else:
                    missing.append(keyword)
        
        return matched, missing
    
    def _calculate_coherence(self, answer: str) -> float:
        """Calculate coherence score based on answer structure and flow."""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        score = 0.5  # Base score
        
        # Check for sentence structure
        sentences = answer.split('.')
        if len(sentences) >= 2:
            score += 0.2
        
        # Check for reasonable length (not too short, not too long)
        word_count = len(answer.split())
        if 20 <= word_count <= 500:
            score += 0.2
        elif word_count < 20:
            score -= 0.1
        
        # Check for technical terms (indicates substance)
        technical_indicators = ['because', 'example', 'implement', 'design', 'algorithm', 'system']
        if any(indicator in answer.lower() for indicator in technical_indicators):
            score += 0.1
        
        return max(0.0, min(1.0, score))
    
    def _calculate_completeness(self, answer: str, question: str) -> float:
        """Calculate how completely the answer addresses the question."""
        if not answer or len(answer.strip()) < 10:
            return 0.0
        
        # Simple heuristic: longer answers that mention question keywords are more complete
        question_keywords = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Check keyword overlap
        overlap = len(question_keywords & answer_words)
        keyword_score = min(1.0, overlap / max(1, len(question_keywords) * 0.3))
        
        # Length score (answers should be substantial but not excessive)
        word_count = len(answer.split())
        if word_count < 20:
            length_score = word_count / 20.0
        elif word_count <= 300:
            length_score = 1.0
        else:
            length_score = max(0.5, 1.0 - (word_count - 300) / 500.0)
        
        return (keyword_score * 0.6 + length_score * 0.4)
    
    def _generate_feedback(self, results: Dict) -> List[str]:
        """Generate feedback based on evaluation results."""
        feedback = []
        
        if results["overall_score"] >= 0.8:
            feedback.append("Excellent answer! The response demonstrates strong understanding.")
        elif results["overall_score"] >= 0.6:
            feedback.append("Good answer with room for improvement.")
        elif results["overall_score"] >= 0.4:
            feedback.append("The answer needs more detail and specificity.")
        else:
            feedback.append("The answer requires significant improvement.")
        
        if results["skill_coverage"] < 0.5:
            feedback.append(f"Consider mentioning more relevant skills. Only {len(results['extracted_skills'])} skills were identified.")
        
        if results["keyword_match"] < 0.5 and results.get("missing_keywords"):
            feedback.append(f"Missing important keywords: {', '.join(results['missing_keywords'][:5])}")
        
        if results["coherence_score"] < 0.5:
            feedback.append("The answer could be more structured and coherent.")
        
        if results["completeness_score"] < 0.5:
            feedback.append("The answer doesn't fully address all aspects of the question.")
        
        return feedback
    
    def _generate_suggestions(self, results: Dict, answer: str) -> List[str]:
        """Generate improvement suggestions."""
        suggestions = []
        
        if len(answer.split()) < 50:
            suggestions.append("Expand your answer with more details and examples.")
        
        if results["skill_coverage"] < 0.7:
            suggestions.append("Include more specific technical skills and technologies in your response.")
        
        if results.get("missing_keywords"):
            suggestions.append(f"Try to incorporate these concepts: {', '.join(results['missing_keywords'][:3])}")
        
        if results["coherence_score"] < 0.6:
            suggestions.append("Structure your answer with clear points: problem, approach, solution, and results.")
        
        if results["semantic_similarity"] < 0.5:
            suggestions.append("Relate your answer more directly to the job requirements and your experience.")
        
        return suggestions
    
    def batch_evaluate(
        self,
        answers: List[str],
        questions: List[str],
        expected_keywords_list: Optional[List[List[str]]] = None,
        expected_skills_list: Optional[List[List[str]]] = None,
        context: Optional[str] = None
    ) -> List[Dict]:
        """
        Evaluate multiple answers in batch.
        
        Args:
            answers: List of candidate answers
            questions: List of corresponding questions
            expected_keywords_list: Optional list of expected keywords for each question
            expected_skills_list: Optional list of expected skills for each question
            context: Optional shared context
        
        Returns:
            List of evaluation results
        """
        results = []
        for i, (answer, question) in enumerate(zip(answers, questions)):
            expected_keywords = expected_keywords_list[i] if expected_keywords_list else None
            expected_skills = expected_skills_list[i] if expected_skills_list else None
            
            eval_result = self.evaluate_answer(
                answer, question, expected_keywords, expected_skills, context
            )
            results.append(eval_result)
        
        return results

