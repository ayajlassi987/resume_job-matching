"""Build a FAISS index of interview questions using OpenAI embeddings (text-embedding-3-small)

Place this script at: `scripts/build_interview_index.py` in the project root.

Usage examples:
  # basic (uses OPENAI_API_KEY when available, else falls back to sentence-transformers)
  python scripts/build_interview_index.py --questions-file data/interview_questions.txt --index-dir vector_store/faiss_index --verify

  # choose model and write index files (overwrite existing)
  python scripts/build_interview_index.py --questions-file data/interview_questions.txt --index-dir vector_store/faiss_index --model text-embedding-3-small --overwrite --top-k 5

What it does:
  - Loads questions from a newline-separated file
  - Generates embeddings (OpenAI if KEY present, else local SentenceTransformer fallback)
  - Builds a FAISS Index (IndexFlatIP on normalized embeddings for cosine similarity)
  - Persists index file and a `meta.json` with question list
  - Optionally verifies by running a small query and printing top matches

Produces (in the target `index-dir`):
  - index.faiss    (the FAISS index)
  - embeddings.npy (raw embeddings)
  - meta.json      (list of questions)

This script is designed to be GitHub-ready and defensible: it logs fallback behavior,
handles missing keys, and is idempotent when `--overwrite` is provided.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

import sys
from pathlib import Path

# Make project importable when running the script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_engine import EmbeddingEngine


def load_questions(path: Path) -> List[str]:
    txt = path.read_text(encoding="utf-8")
    lines = [l.strip() for l in txt.splitlines() if l.strip()]
    return lines


def embed_with_openai(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    if openai is None:
        raise RuntimeError("openai package not installed")
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai.api_key = key
    out = []
    B = 100  # batch
    for i in range(0, len(texts), B):
        batch = texts[i : i + B]
        resp = openai.Embedding.create(model=model, input=batch)
        for r in resp.data:
            out.append(r.embedding)
    arr = np.asarray(out, dtype=np.float32)
    return arr


def embed_with_fallback(texts: List[str]) -> np.ndarray:
    # local fallback using sentence-transformers (same fallback used in project)
    try:
        from sentence_transformers import SentenceTransformer

        m = SentenceTransformer("all-MiniLM-L6-v2")
        emb = m.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)
    except Exception as e:
        raise RuntimeError("No embedding method available. Install openai or sentence-transformers.")


def normalize_embeddings(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def build_faiss_index(emb: np.ndarray, index_path: Path):
    if faiss is None:
        raise RuntimeError("faiss is not installed or failed to import")
    d = emb.shape[1]
    # we expect cosine similarity, so normalize and use inner-product index
    faiss.normalize_L2(emb)
    idx = faiss.IndexFlatIP(d)
    idx.add(emb)
    faiss.write_index(idx, str(index_path / "index.faiss"))


def save_metadata(questions: List[str], emb: np.ndarray, index_dir: Path):
    np.save(index_dir / "embeddings.npy", emb)
    with open(index_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)


def verify_index(query: str, index_dir: Path, top_k: int = 3):
    if faiss is None:
        print("Skipping verification: faiss not available")
        return
    idx_file = index_dir / "index.faiss"
    if not idx_file.exists():
        print("Index file not found for verification")
        return
    idx = faiss.read_index(str(idx_file))
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    questions = meta.get("questions", [])
    # load embeddings to get dimension
    emb = np.load(index_dir / "embeddings.npy")
    # embed the query using same method as saved index: try OpenAI then fallback
    try:
        q_emb = embed_with_openai([query])
    except Exception:
        q_emb = embed_with_fallback([query])
    faiss.normalize_L2(q_emb)
    D, I = idx.search(q_emb, top_k)
    print(f"Verification results for query: {query}")
    for rank, (i, score) in enumerate(zip(I[0], D[0]), start=1):
        print(f"{rank}. score={float(score):.4f} - {questions[i]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions-file", required=True, help="Path to newline-separated interview questions")
    parser.add_argument("--index-dir", default="vector_store/faiss_index", help="Directory to write index files")
    parser.add_argument("--model", default="text-embedding-3-small", help="OpenAI embedding model to use")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing index files")
    parser.add_argument("--verify", action="store_true", help="Run a verification query after building")
    parser.add_argument("--verify-query", default="Tell me about a project where you used Python", help="Query to use for verification")
    parser.add_argument("--top-k", type=int, default=3, help="Top K for verification search")
    args = parser.parse_args()

    qpath = Path(args.questions_file)
    if not qpath.exists():
        raise SystemExit(f"Questions file not found: {qpath}")
    questions = load_questions(qpath)
    if not questions:
        raise SystemExit("No questions found in the input file")

    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    # if index exists and not overwrite, bail out
    if any((index_dir / fname).exists() for fname in ("index.faiss", "embeddings.npy", "meta.json")) and not args.overwrite:
        print("Index files already exist in index dir. Use --overwrite to replace them.")
        return

    # Build index using EmbeddingEngine (it will prefer OpenAI if configured)
    engine = EmbeddingEngine(model=args.model, index_dir=str(index_dir))
    try:
        engine.build_index(questions)
    except Exception as e:
        raise SystemExit(f"Failed to build index: {e}")

    print(f"Index built and saved to {index_dir} (n={len(questions)}) - model={engine.model} use_openai={engine.use_openai}")

    if args.verify:
        results = engine.search(args.verify_query, top_k=args.top_k)
        print(f"Verification results for query: {args.verify_query}")
        for rank, (job, score) in enumerate(results, start=1):
            print(f"{rank}. score={score:.4f} - {job}")


if __name__ == "__main__":
    main()
