"""Monitoring script for system health and performance."""
import time
import json
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.embedding_engine import EmbeddingEngine
from src.rag_engine import RAGEngine
from src.local_llm import ask_llm
from config import VECTOR_STORE_DIR


def check_vector_store() -> Dict[str, Any]:
    """Check vector store health."""
    try:
        emb = EmbeddingEngine(index_dir=str(VECTOR_STORE_DIR))
        emb.load_index()
        
        if not emb._docs:
            return {"status": "error", "message": "No documents in index"}
        
        # Test search
        start = time.time()
        results = emb.search("test query", top_k=3)
        search_time = time.time() - start
        
        return {
            "status": "ok",
            "document_count": len(emb._docs),
            "search_time_ms": search_time * 1000,
            "index_loaded": emb._index is not None
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_llm() -> Dict[str, Any]:
    """Check LLM availability."""
    try:
        start = time.time()
        response = ask_llm("Say 'OK' if you are working.", model="mistral")
        llm_time = time.time() - start
        
        return {
            "status": "ok",
            "response_time_ms": llm_time * 1000,
            "response_received": bool(response)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def check_rag_pipeline() -> Dict[str, Any]:
    """Check RAG pipeline end-to-end."""
    try:
        rag = RAGEngine()
        
        test_resume = "Software engineer with 5 years of experience in Python and machine learning."
        test_job = "Looking for a senior Python developer with ML experience."
        
        start = time.time()
        questions = rag.generate_questions(test_resume, test_job, top_k=5)
        pipeline_time = time.time() - start
        
        return {
            "status": "ok",
            "pipeline_time_ms": pipeline_time * 1000,
            "questions_generated": len(questions),
            "questions_valid": all(len(q) > 10 for q in questions)
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_system_stats() -> Dict[str, Any]:
    """Get overall system statistics."""
    stats = {
        "timestamp": time.time(),
        "components": {
            "vector_store": check_vector_store(),
            "llm": check_llm(),
            "rag_pipeline": check_rag_pipeline()
        }
    }
    
    # Calculate overall health
    component_statuses = [c["status"] for c in stats["components"].values()]
    if all(s == "ok" for s in component_statuses):
        stats["overall_status"] = "healthy"
    elif any(s == "error" for s in component_statuses):
        stats["overall_status"] = "degraded"
    else:
        stats["overall_status"] = "warning"
    
    return stats


def main():
    """Run monitoring checks."""
    print("Running system health checks...")
    stats = get_system_stats()
    
    print("\n" + "="*50)
    print("System Health Report")
    print("="*50)
    print(f"Overall Status: {stats['overall_status'].upper()}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(stats['timestamp']))}")
    print("\nComponent Status:")
    
    for component, status in stats["components"].items():
        print(f"\n  {component.upper()}:")
        print(f"    Status: {status['status']}")
        if status["status"] == "ok":
            for key, value in status.items():
                if key != "status":
                    print(f"    {key}: {value}")
        else:
            print(f"    Error: {status.get('message', 'Unknown error')}")
    
    # Output JSON for programmatic use
    print("\n" + "="*50)
    print("JSON Output:")
    print(json.dumps(stats, indent=2))
    
    # Exit with error code if system is not healthy
    if stats["overall_status"] != "healthy":
        sys.exit(1)


if __name__ == "__main__":
    main()

