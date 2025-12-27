"""Retrieval-Augmented Generation (RAG) utilities.

This module provides a `RAGEngine` class which retrieves relevant interview
questions from a FAISS-backed index and uses an LLM to generate tailored
interview questions following the project's RAG prompt structure.

Supports enhanced query construction with skill-based expansion and
different question types (technical, behavioral, system design).
"""
from typing import List, Optional, Dict
from src.embedding_engine import EmbeddingEngine
from src.local_llm import ask_llm
from src.skill_extractor import extract_skills
from src.chunking import manage_context_window


class RAGEngine:
    def __init__(self, index_dir="vector_store/faiss_index"):
        # Force local embeddings - disable OpenAI
        self.emb = EmbeddingEngine(index_dir=index_dir, use_openai=False)
        self.index_dir = index_dir
        self.prompt_templates = self._load_prompt_templates()
    
    def _load_prompt_templates(self) -> Dict[str, str]:
        """Load prompt templates for different question types."""
        return {
            "technical": """You are a professional technical interviewer specializing in technical assessments.

Candidate Resume:
{resume}

Job Description:
{job}

Relevant Interview Questions:
{retrieved_questions}

Generate 5 tailored technical interview questions that assess:
- Programming skills and problem-solving abilities
- Technical knowledge relevant to the role
- Hands-on experience with specific technologies
- Code quality and best practices

Format each question clearly and make them specific to the candidate's background and the job requirements.""",
            
            "behavioral": """You are a professional interviewer specializing in behavioral assessments.

Candidate Resume:
{resume}

Job Description:
{job}

Relevant Interview Questions:
{retrieved_questions}

Generate 5 tailored behavioral interview questions that assess:
- Past experiences and how they handle situations
- Leadership and teamwork abilities
- Problem-solving approach
- Cultural fit and work style

Format each question clearly and make them relevant to the candidate's experience and the job requirements.""",
            
            "system_design": """You are a senior technical interviewer specializing in system design assessments.

Candidate Resume:
{resume}

Job Description:
{job}

Relevant Interview Questions:
{retrieved_questions}

Generate 5 tailored system design interview questions that assess:
- Architecture and design thinking
- Scalability and performance considerations
- Trade-off analysis
- Real-world system design experience

Format each question clearly and make them appropriate for the candidate's level and the role requirements.""",
            
            "default": """You are a professional interviewer.

Candidate Resume:
{resume}

Job Description:
{job}

Relevant Interview Questions:
{retrieved_questions}

Generate 5 tailored interview questions that are relevant to the candidate's background and the job requirements."""
        }
    
    def _expand_query_with_skills(self, resume: str, job: str) -> str:
        """Expand query with extracted skills for better retrieval."""
        resume_skills = extract_skills(resume)
        job_skills = extract_skills(job)
        
        # Combine unique skills
        all_skills = list(set(resume_skills + job_skills))
        
        # Build enhanced query
        base_query = f"{resume}\n{job}"
        if all_skills:
            skills_text = ", ".join(all_skills[:10])  # Limit to top 10 skills
            enhanced_query = f"{base_query}\n\nKey Skills: {skills_text}"
            return enhanced_query
        
        return base_query
    
    def _decompose_query(self, resume: str, job: str) -> List[str]:
        """Decompose complex query into multiple focused queries."""
        queries = []
        
        # Query 1: Resume-focused
        resume_skills = extract_skills(resume)
        if resume_skills:
            queries.append(f"Interview questions about {', '.join(resume_skills[:5])}")
        
        # Query 2: Job-focused
        job_skills = extract_skills(job)
        if job_skills:
            queries.append(f"Interview questions for {', '.join(job_skills[:5])}")
        
        # Query 3: Combined
        queries.append(f"{resume}\n{job}")
        
        return queries
    
    def _retrieve_with_multiple_queries(
        self,
        resume: str,
        job: str,
        top_k: int = 8,
        use_query_decomposition: bool = False
    ) -> List[tuple]:
        """Retrieve questions using multiple query strategies."""
        if use_query_decomposition:
            queries = self._decompose_query(resume, job)
            all_results = []
            seen_questions = set()
            
            for query in queries:
                results = self.emb.search(query, top_k=top_k // len(queries) + 1)
                for q, score in results:
                    if q not in seen_questions:
                        all_results.append((q, score))
                        seen_questions.add(q)
            
            # Sort by score and return top-k
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results[:top_k]
        else:
            # Use skill-expanded query
            enhanced_query = self._expand_query_with_skills(resume, job)
            return self.emb.search(enhanced_query, top_k=top_k)
    
    def generate_questions(
        self,
        resume: str,
        job: str,
        top_k: int = 8,
        question_type: str = "default",
        use_query_decomposition: bool = False,
        use_hybrid_search: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[str]:
        """
        Generate tailored interview questions using RAG.
        
        Args:
            resume: Candidate resume text
            job: Job description text
            top_k: Number of retrieved questions to use
            question_type: Type of questions ("technical", "behavioral", "system_design", "default")
            use_query_decomposition: Use query decomposition strategy
            use_hybrid_search: Override hybrid search setting
            use_reranking: Override reranking setting
        
        Returns:
            List of generated interview questions
        """
        # Retrieve relevant questions
        retrieved = self._retrieve_with_multiple_queries(
            resume, job, top_k=top_k * 2, use_query_decomposition=use_query_decomposition
        )
        
        # If hybrid search or reranking is explicitly requested, re-retrieve
        if use_hybrid_search is not None or use_reranking is not None:
            enhanced_query = self._expand_query_with_skills(resume, job)
            retrieved = self.emb.search(
                enhanced_query,
                top_k=top_k * 2,
                use_hybrid=use_hybrid_search,
                use_reranking=use_reranking
            )
        
        retrieved_text = "\n".join([f"- {q}" for q, _ in retrieved[:top_k]])
        
        # Get appropriate prompt template
        template = self.prompt_templates.get(question_type, self.prompt_templates["default"])
        
        # Manage context window
        resume_chunks = manage_context_window([resume], max_tokens=1500)
        job_chunks = manage_context_window([job], max_tokens=1500)
        resume_text = resume_chunks[0] if resume_chunks else resume[:2000]
        job_text = job_chunks[0] if job_chunks else job[:2000]
        
        prompt = template.format(
            resume=resume_text,
            job=job_text,
            retrieved_questions=retrieved_text
        )
        
        output = ask_llm(prompt)
        
        return self._parse(output)
    
    def _parse(self, text: str) -> List[str]:
        """Parse LLM output to extract questions."""
        lines = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            # Remove common prefixes and numbering
            line = line.lstrip("-•0123456789. ")
            # Remove markdown formatting
            if line.startswith("#"):
                continue
            if line and len(line) > 10:  # Filter out very short lines
                lines.append(line)
        
        # Return top 5 questions
        return lines[:5]


