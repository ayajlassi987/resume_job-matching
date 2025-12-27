"""Embedding engine with OpenAI integration and FAISS persistence.

This module chooses OpenAI embeddings (`text-embedding-3-small`) when
`OPENAI_API_KEY` is present (and the `openai` package is available). Otherwise
it falls back to `sentence-transformers` (`all-MiniLM-L6-v2`) locally.

It provides helpers to build a FAISS index and search it.
Supports hybrid search (semantic + BM25) and reranking.
"""
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict

import numpy as np

try:
    import openai
except Exception:
    openai = None

try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

try:
    from sentence_transformers.cross_encoder import CrossEncoder
except Exception:
    CrossEncoder = None


class EmbeddingEngine:
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        index_dir: str = "vector_store/faiss_index",
        use_openai: bool = None,
        enable_hybrid_search: bool = True,
        enable_reranking: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """Create an embedding engine.

        If `use_openai` is None, the engine will use OpenAI if `OPENAI_API_KEY` is set.
        
        Args:
            model: Embedding model name
            index_dir: Directory for FAISS index files
            use_openai: Force OpenAI usage (None = auto-detect)
            enable_hybrid_search: Enable BM25 hybrid search
            enable_reranking: Enable cross-encoder reranking
            reranker_model: Cross-encoder model for reranking
        """
        self.model = model
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index = None
        self._docs: List[str] = []
        self._bm25: Optional[BM25Okapi] = None
        self._reranker: Optional[CrossEncoder] = None
        self.enable_hybrid_search = enable_hybrid_search and BM25Okapi is not None
        self.enable_reranking = enable_reranking and CrossEncoder is not None
        
        if use_openai is None:
            # Default to False - use local embeddings only
            # Only use OpenAI if explicitly enabled via environment variable
            self.use_openai = os.environ.get("ENABLE_OPENAI_EMBEDDINGS", "False").lower() == "true"
        else:
            self.use_openai = bool(use_openai)
        
        # Track if we've already warned about OpenAI failures
        self._openai_warned = False
        
        if self.enable_reranking:
            try:
                self._reranker = CrossEncoder(reranker_model)
            except Exception as e:
                print(f"Warning: Could not load reranker model {reranker_model}: {e}")
                self.enable_reranking = False

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        if openai is None:
            raise RuntimeError("openai package is not installed")
        
        # Check if OpenAI is disabled due to quota error
        if os.environ.get("DISABLE_OPENAI_EMBEDDINGS", "").lower() == "true":
            raise RuntimeError("OpenAI embeddings disabled due to quota error")
        
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        openai.api_key = key
        out = []
        batch = 100
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            # Support both old (openai.Embedding.create) and new (OpenAI client) libraries
            # Prefer new-style client when available to support openai>=1.0.0
            try:
                client_cls = getattr(openai, "OpenAI", None)
                if client_cls is not None:
                    client = client_cls()
                    resp = client.embeddings.create(model=self.model, input=chunk)
                else:
                    # Fallback to legacy API (older openai package versions)
                    resp = openai.Embedding.create(input=chunk, model=self.model)
            except Exception:
                # Let caller handle fallback to local model; include message for visibility
                raise
            for item in resp.data:
                # item may be a dict or an object with .embedding
                if isinstance(item, dict):
                    out.append(item.get("embedding"))
                else:
                    out.append(getattr(item, "embedding", None) or getattr(item, "vector", None))
        return np.asarray(out, dtype=np.float32)

    def _embed_local(self, texts: List[str]) -> np.ndarray:
        """Embed texts using local SentenceTransformer model (legacy method)."""
        return self._embed_local_batch(texts, batch_size=100)

    def encode(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """
        Return embeddings for a list of texts as float32 numpy array.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (for local models)
        
        Returns:
            Numpy array of embeddings
        """
        # Always use local embeddings - OpenAI is disabled by default
        if self.use_openai:
            # Only try OpenAI if explicitly enabled, but still fallback to local
            try:
                return self._embed_openai(texts)
            except Exception as e:
                # Silently fallback to local - no error messages
                self.use_openai = False
                return self._embed_local_batch(texts, batch_size)
        else:
            return self._embed_local_batch(texts, batch_size)
    
    def _embed_local_batch(self, texts: List[str], batch_size: int = 100) -> np.ndarray:
        """Embed texts in batches for better performance."""
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers is not installed")
        
        # Load model once (could be cached)
        if not hasattr(self, '_local_model'):
            self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            emb = self._local_model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
            all_embeddings.append(emb)
        
        return np.vstack(all_embeddings).astype(np.float32)

    def build_index(
        self,
        docs: List[str],
        index_type: str = "flat",
        nlist: int = 100,
        nprobe: int = 10
    ):
        """
        Build and persist a FAISS index for the provided docs.
        
        Args:
            docs: List of documents to index
            index_type: Type of index ("flat", "ivf", "hnsw")
            nlist: Number of clusters for IVF index
            nprobe: Number of clusters to probe for IVF search
        """
        self._docs = docs
        emb = self.encode(docs)
        if faiss is None:
            raise RuntimeError("faiss is not installed")
        dim = emb.shape[1]
        faiss.normalize_L2(emb)
        
        # Choose index type based on dataset size
        num_docs = len(docs)
        
        if index_type == "flat" or num_docs < 1000:
            # Use flat index for small datasets
            index = faiss.IndexFlatIP(dim)
        elif index_type == "ivf":
            # Use IVF index for medium to large datasets
            quantizer = faiss.IndexFlatIP(dim)
            index = faiss.IndexIVFFlat(quantizer, dim, min(nlist, num_docs // 10))
            index.train(emb)
        elif index_type == "hnsw":
            # Use HNSW index for very large datasets
            index = faiss.IndexHNSWFlat(dim, 32)  # 32 is the M parameter
        else:
            index = faiss.IndexFlatIP(dim)
        
        index.add(emb)
        
        # Store index type and parameters
        index_params = {
            "index_type": index_type,
            "nlist": nlist if index_type == "ivf" else None,
            "nprobe": nprobe if index_type == "ivf" else None
        }
        
        faiss.write_index(index, str(self.index_dir / "index.faiss"))
        np.save(self.index_dir / "embeddings.npy", emb)
        
        # Build BM25 index for hybrid search
        if self.enable_hybrid_search:
            tokenized_docs = [doc.lower().split() for doc in docs]
            self._bm25 = BM25Okapi(tokenized_docs)
        
        with open(self.index_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({
                "docs": docs,
                "model": self.model,
                "use_openai": self.use_openai,
                "enable_hybrid_search": self.enable_hybrid_search,
                "enable_reranking": self.enable_reranking
            }, f, ensure_ascii=False, indent=2)
        self._index = index

    def load_index(self):
        meta_path = self.index_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
                self._docs = meta.get("docs", [])
                # Restore BM25 index if hybrid search was enabled
                if self.enable_hybrid_search and self._docs:
                    tokenized_docs = [doc.lower().split() for doc in self._docs]
                    self._bm25 = BM25Okapi(tokenized_docs)
        if faiss is not None and (self.index_dir / "index.faiss").exists():
            self._index = faiss.read_index(str(self.index_dir / "index.faiss"))

    def _semantic_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Perform semantic search using embeddings."""
        q_emb = self.encode([query])
        if faiss is not None and self._index is not None:
            faiss.normalize_L2(q_emb)
            
            # Handle different index types
            if isinstance(self._index, faiss.IndexIVFFlat):
                # Set nprobe for IVF index
                self._index.nprobe = min(10, self._index.ntotal // 10) if self._index.ntotal > 0 else 10
            
            D, I = self._index.search(q_emb, top_k * 2)  # Get more candidates for reranking
            results = []
            for idx, score in zip(I[0], D[0]):
                if idx >= 0:  # Valid index
                    results.append((int(idx), float(score)))
            return results
        else:
            emb = np.load(self.index_dir / "embeddings.npy")
            def normalize(x):
                norms = np.linalg.norm(x, axis=1, keepdims=True)
                norms[norms == 0] = 1
                return x / norms
            emb_norm = normalize(emb)
            q_norm = normalize(q_emb)
            sims = (emb_norm @ q_norm.T).squeeze()
            idxs = np.argsort(-sims)[:top_k * 2]
            return [(int(i), float(sims[i])) for i in idxs]
    
    def _bm25_search(self, query: str, top_k: int = 3) -> List[Tuple[int, float]]:
        """Perform BM25 keyword search."""
        if not self._bm25:
            return []
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k * 2]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
    
    def _hybrid_search(self, query: str, top_k: int = 3, alpha: float = 0.7) -> List[Tuple[int, float]]:
        """Combine semantic and BM25 search results."""
        semantic_results = self._semantic_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        
        # Normalize scores to [0, 1] range
        def normalize_scores(results: List[Tuple[int, float]]) -> Dict[int, float]:
            if not results:
                return {}
            max_score = max(score for _, score in results) if results else 1.0
            min_score = min(score for _, score in results) if results else 0.0
            score_range = max_score - min_score if max_score != min_score else 1.0
            return {idx: (score - min_score) / score_range for idx, score in results}
        
        semantic_scores = normalize_scores(semantic_results)
        bm25_scores = normalize_scores(bm25_results)
        
        # Combine scores
        combined_scores = {}
        all_indices = set(semantic_scores.keys()) | set(bm25_scores.keys())
        
        for idx in all_indices:
            sem_score = semantic_scores.get(idx, 0.0)
            bm25_score = bm25_scores.get(idx, 0.0)
            combined_scores[idx] = alpha * sem_score + (1 - alpha) * bm25_score
        
        # Sort by combined score
        sorted_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in sorted_results[:top_k * 2]]
    
    def _rerank(self, query: str, candidates: List[Tuple[int, float]], top_k: int = 3) -> List[Tuple[str, float]]:
        """Rerank candidates using cross-encoder."""
        if not self._reranker or not candidates:
            # Return original results without reranking
            return [(self._docs[idx], score) for idx, score in candidates[:top_k]]
        
        # Prepare pairs for reranking
        pairs = [(query, self._docs[idx]) for idx, _ in candidates]
        rerank_scores = self._reranker.predict(pairs)
        
        # Combine original scores with rerank scores (weighted)
        reranked = []
        for i, (idx, orig_score) in enumerate(candidates):
            rerank_score = float(rerank_scores[i])
            # Weighted combination: 30% original, 70% rerank
            final_score = 0.3 * orig_score + 0.7 * rerank_score
            reranked.append((idx, final_score))
        
        # Sort by final score
        reranked.sort(key=lambda x: x[1], reverse=True)
        
        return [(self._docs[idx], score) for idx, score in reranked[:top_k]]
    
    def search(
        self,
        query: str,
        top_k: int = 3,
        use_hybrid: Optional[bool] = None,
        use_reranking: Optional[bool] = None
    ) -> List[Tuple[str, float]]:
        """
        Search the persisted index for `query` and return top-k (doc, score) pairs.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_hybrid: Override hybrid search setting
            use_reranking: Override reranking setting
        
        Returns:
            List of (document, score) tuples
        """
        if not self._docs:
            self.load_index()
        
        use_hybrid = use_hybrid if use_hybrid is not None else self.enable_hybrid_search
        use_reranking = use_reranking if use_reranking is not None else self.enable_reranking
        
        # Perform search
        if use_hybrid:
            candidates = self._hybrid_search(query, top_k)
        else:
            candidates = self._semantic_search(query, top_k)
        
        # Rerank if enabled
        if use_reranking and len(candidates) > 1:
            return self._rerank(query, candidates, top_k)
        else:
            # Return top-k without reranking
            return [(self._docs[idx], score) for idx, score in candidates[:top_k]]
