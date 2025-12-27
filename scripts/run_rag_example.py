"""Example script demonstrating RAG: retrieve questions and generate tailored ones."""
import sys
from pathlib import Path

# Make project importable when running directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag_engine import RAGEngine


def main():
    rag = RAGEngine(index_dir="vector_store/faiss_index")

    # Load a sample resume and job description from `data/` if present
    resume_path = ROOT / "data" / "resumes" / "resume_0.txt"
    job_path = ROOT / "data" / "job_descriptions" / "jd_92461.txt"
    resume = resume_path.read_text(encoding="utf-8") if resume_path.exists() else "Experienced Python developer with Docker and AWS"
    job = job_path.read_text(encoding="utf-8") if job_path.exists() else "Backend Software Engineer: Python, REST APIs, PostgreSQL, Docker"

    # Retrieve questions
    retrieved = rag.emb.search(rag._expand_query_with_skills(resume, job), top_k=10)
    print("Retrieved candidate questions (top 10):")
    for q, s in retrieved:
        print(f"- score={s:.4f}: {q}")

    print("\nGenerating 5 tailored interview questions using RAG prompt...")
    questions = rag.generate_questions(resume, job, top_k=8, question_type="technical")
    print("\nGenerated Questions:")
    for i, q in enumerate(questions, start=1):
        print(f"{i}. {q}")


if __name__ == "__main__":
    main()
