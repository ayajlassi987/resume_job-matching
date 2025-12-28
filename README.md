# AI-Resume-Interview-Copilot

## Overview
An AI-powered system that analyzes resumes, matches them with job descriptions,
identifies skill gaps, and generates tailored interview questions using RAG (Retrieval-Augmented Generation).

This project implements a Retrieval-Augmented Generation (RAG) system for interview question generation. It combines semantic retrieval using FAISS with a locally deployed open-source LLM, ensuring grounded, domain-specific question generation without reliance on paid APIs.

Given resumes and job descriptions, the system:

			Matches candidates to jobs

			Retrieves relevant interview questions from a dataset (RAG)

			Uses a local open-source LLM to generate tailored interview questions


## Features

### Core Functionality
- **Resume Parsing**: Support for PDF, DOCX, and TXT formats with structured section extraction
- **Enhanced Skill Extraction**: NER-based extraction with fuzzy matching and skill taxonomy
- **Semantic Resume–Job Matching**: TF-IDF and embedding-based matching
- **RAG-based Question Generation**: 
  - Hybrid search (semantic + BM25) for better retrieval
  - Cross-encoder reranking for improved relevance
  - Multiple question types (technical, behavioral, system design)
  - Skill-based query expansion
- **Answer Evaluation**: Comprehensive evaluation with semantic similarity, skill coverage, and feedback generation

### Advanced Features
- **Multiple LLM Backends**: Support for Ollama, Hugging Face Transformers, llama.cpp, and vLLM
- **REST API**: Full Flask API with request validation and error handling
- **Performance Optimizations**: 
  - Efficient FAISS indices (IVF, HNSW) for large datasets
  - Batch processing for embeddings
- **Monitoring & Logging**: Structured JSON logging and health monitoring
- **Docker Support**: Containerized deployment with Docker Compose

## Tech Stack
- **Python 3.10+**
- **Vector Search**: FAISS (with IVF/HNSW support)
- **Embeddings**: Sentence-Transformers, OpenAI (optional)
- **LLM**: Ollama, Hugging Face Transformers, llama.cpp, vLLM
- **RAG**: Hybrid search (BM25 + semantic), Cross-encoder reranking
- **API**: Flask, Pydantic, Flask-CORS
- **Testing**: pytest, pytest-cov
- **Deployment**: Docker, Docker Compose

## Quick Start

### Local Development

1. **Create virtual environment and install dependencies:**

	```powershell
	# Windows PowerShell
	python -m venv .venv
	.\.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

	```bash
	# Linux/Mac
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

2. **Download spaCy model:**

	```bash
	python -m spacy download en_core_web_sm
	```

3. **Build the interview question FAISS index:**

	```bash
	python scripts/build_interview_index.py \
	  --questions-file data/interview_questions/questions.txt \
	  --index-dir vector_store/faiss_index \
	  --overwrite \
	  --verify
	```

4. **Run the API:**

	```bash
	# Option 1: Run directly (simplest)
	python api/app.py
	```

	Or using Flask CLI:

	```bash
	# Option 2: Using Flask CLI
	# Windows PowerShell
	$env:FLASK_APP="api.app"
	python -m flask run --host=0.0.0.0 --port=5000

	# Linux/Mac
	export FLASK_APP=api.app
	python -m flask run --host=0.0.0.0 --port=5000
	```

5. **Run the Streamlit Frontend:**

	In a new terminal, start the Streamlit app:

	```bash
	streamlit run frontend/app.py
	```

	The frontend will open in your browser at `http://localhost:8501`

6. **Test the API (optional):**

	```bash
	curl http://localhost:5000/api/health
	```

### Docker Deployment

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f api

# Stop services
docker-compose down
```

### Using the API

See [API Documentation](docs/API.md) for detailed endpoint documentation.

**Example: Generate Interview Questions**

```bash
curl -X POST http://localhost:5000/api/generate-questions \
  -H "Content-Type: application/json" \
  -d '{
    "resume": "Python developer with 5 years of ML experience...",
    "job_description": "Looking for a Python ML engineer...",
    "question_type": "technical"
  }'
```

## Configuration

Create a `.env` file for configuration:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False

# LLM Configuration
LLM_BACKEND=ollama  # Options: ollama, huggingface, llama_cpp, vllm
LLM_MODEL=mistral

# Embedding Configuration
OPENAI_API_KEY=your_key_here  # Optional
EMBEDDING_MODEL=text-embedding-3-small
DISABLE_OPENAI_EMBEDDINGS=False  # Set to True to force local embeddings only (useful if quota exceeded)

# RAG Configuration
RAG_TOP_K=8
ENABLE_HYBRID_SEARCH=True
ENABLE_RERANKING=True

```

See [Deployment Guide](docs/DEPLOYMENT.md) for more configuration options.

## Architecture

See `docs/ARCHITECTURE.md` and `docs/architecture.puml` for the system diagram and details.

### System Flow

1. **Resume Processing**: Parse and extract structured information (sections, skills, contact info)
2. **Job Matching**: Match resumes to job descriptions using semantic similarity
3. **Question Retrieval**: Use hybrid search (semantic + BM25) to retrieve relevant questions from FAISS index
4. **Reranking**: Apply cross-encoder reranking for improved relevance
5. **Question Generation**: Use local LLM with RAG prompt to generate tailored questions
6. **Answer Evaluation**: Evaluate candidate answers with multiple metrics and generate feedback

### RAG Prompt Templates

The system supports multiple prompt templates for different question types:

- **Technical**: Focuses on programming skills, technical knowledge, and hands-on experience
- **Behavioral**: Assesses past experiences, leadership, and teamwork
- **System Design**: Evaluates architecture and design thinking
- **Default**: General-purpose interview questions

See `src/rag_engine.py` for prompt templates.

## Testing

Run tests with pytest:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=api

# Run specific test file
pytest tests/test_rag_engine.py
```

## Monitoring

Check system health:

```bash
# Health check endpoint
curl http://localhost:5000/api/health

# Run monitoring script
python scripts/monitor.py
```

## Documentation

- [API Documentation](docs/API.md) - Complete API reference
- [Deployment Guide](docs/DEPLOYMENT.md) - Deployment instructions
- [Architecture](docs/ARCHITECTURE.md) - System architecture details

## CI / Reproducibility

A GitHub Actions workflow is included to run tests using local embeddings (no API keys required). See `.github/workflows/ci.yml`.

## Project Structure

```
.
├── api/                 # Flask API application
│   ├── app.py          # Flask app initialization
│   ├── routes.py       # API endpoints
│   └── models.py       # Pydantic request/response models
├── frontend/           # Streamlit frontend
│   ├── app.py          # Main Streamlit app
│   ├── api_client.py   # API client functions
│   └── utils.py        # Utility functions
├── src/                 # Core modules
│   ├── rag_engine.py   # RAG question generation
│   ├── embedding_engine.py  # Embeddings and FAISS
│   ├── local_llm.py    # Local LLM integration
│   ├── answer_evaluator.py  # Answer evaluation
│   ├── resume_parser.py # Resume parsing
│   ├── skill_extractor.py   # Skill extraction
│   └── logger.py       # Structured logging
├── tests/              # Test suite
├── scripts/            # Utility scripts
├── docs/               # Documentation
├── vector_store/       # FAISS indices
└── data/               # Data files
```

## Report

See `REPORT.md` for a project report outlining design decisions, limitations, and next steps.

## License

[Add your license here]
