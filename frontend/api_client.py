"""API client for interacting with the Interview Copilot Flask API."""
import requests
from typing import List, Dict, Optional, Any
import os


class APIClient:
    """Client for making requests to the Interview Copilot API."""
    
    def __init__(self, base_url: str = None):
        """
        Initialize API client.
        
        Args:
            base_url: Base URL of the API (default: http://localhost:5000)
        """
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:5000")
        self.timeout = 600  # 10 minute timeout (for LLM generation)
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check API health status.
        
        Returns:
            Dictionary with health status information
        """
        try:
            response = requests.get(f"{self.base_url}/api/health", timeout=5)
            response.raise_for_status()
            return {"status": "healthy", "data": response.json()}
        except requests.exceptions.ConnectionError:
            return {"status": "error", "message": "Cannot connect to API. Make sure the Flask server is running."}
        except requests.exceptions.Timeout:
            return {"status": "error", "message": "API request timed out."}
        except Exception as e:
            return {"status": "error", "message": f"Error checking API health: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Dictionary with statistics
        """
        try:
            response = requests.get(f"{self.base_url}/api/stats", timeout=self.timeout)
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except Exception as e:
            return {"status": "error", "message": f"Error getting stats: {str(e)}"}
    
    def match_resume(
        self,
        resume: str,
        job_descriptions: List[str],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Match resume to job descriptions.
        
        Args:
            resume: Resume text
            job_descriptions: List of job description texts
            top_k: Number of top matches to return
        
        Returns:
            Dictionary with match results
        """
        try:
            payload = {
                "resume": resume,
                "job_descriptions": job_descriptions,
                "top_k": top_k
            }
            response = requests.post(
                f"{self.base_url}/api/match",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except requests.exceptions.HTTPError as e:
            error_msg = "Unknown error"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
            except:
                error_msg = str(e)
            return {"status": "error", "message": f"API error: {error_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"Error matching resume: {str(e)}"}
    
    def generate_questions(
        self,
        resume: str,
        job_description: str,
        question_type: str = "default",
        top_k: int = 8,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Generate interview questions.
        
        Args:
            resume: Resume text
            job_description: Job description text
            question_type: Type of questions (technical, behavioral, system_design, default)
            top_k: Number of retrieved questions to use
            use_hybrid_search: Override hybrid search setting
            use_reranking: Override reranking setting
        
        Returns:
            Dictionary with generated questions
        """
        try:
            payload = {
                "resume": resume,
                "job_description": job_description,
                "question_type": question_type,
                "top_k": top_k
            }
            if use_hybrid_search is not None:
                payload["use_hybrid_search"] = use_hybrid_search
            if use_reranking is not None:
                payload["use_reranking"] = use_reranking
            
            response = requests.post(
                f"{self.base_url}/api/generate-questions",
                json=payload,
                timeout=self.timeout  # 10 minutes for question generation
            )
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except requests.exceptions.HTTPError as e:
            error_msg = "Unknown error"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
            except:
                error_msg = str(e)
            return {"status": "error", "message": f"API error: {error_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"Error generating questions: {str(e)}"}
    
    def evaluate_answer(
        self,
        answer: str,
        question: str,
        expected_keywords: Optional[List[str]] = None,
        expected_skills: Optional[List[str]] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a candidate's answer.
        
        Args:
            answer: Candidate's answer text
            question: Interview question
            expected_keywords: Expected keywords in answer
            expected_skills: Expected skills to demonstrate
            context: Additional context (resume/job description)
        
        Returns:
            Dictionary with evaluation results
        """
        try:
            payload = {
                "answer": answer,
                "question": question
            }
            if expected_keywords:
                payload["expected_keywords"] = expected_keywords
            if expected_skills:
                payload["expected_skills"] = expected_skills
            if context:
                payload["context"] = context
            
            response = requests.post(
                f"{self.base_url}/api/evaluate-answer",
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            return {"status": "success", "data": response.json()}
        except requests.exceptions.HTTPError as e:
            error_msg = "Unknown error"
            try:
                error_data = e.response.json()
                error_msg = error_data.get("error", str(e))
            except:
                error_msg = str(e)
            return {"status": "error", "message": f"API error: {error_msg}"}
        except Exception as e:
            return {"status": "error", "message": f"Error evaluating answer: {str(e)}"}


# Global client instance
_client = None

def get_client() -> APIClient:
    """Get or create global API client instance."""
    global _client
    if _client is None:
        _client = APIClient()
    return _client

