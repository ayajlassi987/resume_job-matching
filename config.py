"""Configuration management for the application."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store" / "faiss_index"

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 5000))
API_DEBUG = os.getenv("API_DEBUG", "False").lower() == "true"

# LLM Configuration
LLM_BACKEND = os.getenv("LLM_BACKEND", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "mistral")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
USE_OPENAI = bool(OPENAI_API_KEY)

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Disable OpenAI embeddings by default - use local embeddings only
DISABLE_OPENAI_EMBEDDINGS = os.getenv("DISABLE_OPENAI_EMBEDDINGS", "True").lower() == "true"
USE_OPENAI_EMBEDDINGS = False  # Always use local embeddings

# RAG Configuration
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 8))
ENABLE_HYBRID_SEARCH = os.getenv("ENABLE_HYBRID_SEARCH", "True").lower() == "true"
ENABLE_RERANKING = os.getenv("ENABLE_RERANKING", "True").lower() == "true"

# Rate Limiting
RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = os.getenv("LOG_FILE", "app.log")

