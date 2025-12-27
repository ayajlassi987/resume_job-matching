"""CI smoke test: build a tiny index and run a RAG generation using local embeddings.

This script is intended for CI runs and local verification without OpenAI keys.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.embedding_engine import EmbeddingEngine
from src.rag_engine import RAGEngine


def main():
    questions_file = ROOT / "data" / "ci_small_questions.txt"
    index_dir = ROOT / "vector_store" / "faiss_index_ci"

    print("[ci_smoke_test] Building tiny index with local embeddings...")
    emb = EmbeddingEngine(use_openai=False, index_dir=str(index_dir))
    with open(questions_file, "r", encoding="utf-8") as fh:
        docs = [l.strip() for l in fh if (l := l.strip())]

    emb.build_index(docs)
    print("[ci_smoke_test] Index built OK")

    # Run RAG with a small resume + job
    rag = RAGEngine(index_dir=str(index_dir))
    resume = "Experienced data engineer with strong ETL, SQL, and data warehouse skills."
    job = "Looking for a data engineer familiar with ETL optimization, data quality, and cloud warehouses."

    questions = rag.generate_questions(resume, job, top_k=6, question_type="technical")
    if not questions:
        print("[ci_smoke_test] FAILED: No questions generated")
        sys.exit(2)

    print("[ci_smoke_test] Generated questions:")
    for i, q in enumerate(questions, 1):
        print(f"{i}. {q}")

    print("[ci_smoke_test] SUCCESS")


if __name__ == "__main__":
    main()
