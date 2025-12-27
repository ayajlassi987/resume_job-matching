# Project Report — AI-Resume-Interview-Copilot 📋

## Purpose
This project implements a retrieval-augmented generation (RAG) pipeline to produce tailored interview questions from a candidate resume and job description. The main goals are reproducibility, defense against external API outages, and clear prompt determinism.

## Key Design Decisions
- **Embeddings:** Prefer OpenAI `text-embedding-3-small` when an API key is available. Otherwise, fall back to `sentence-transformers` (`all-MiniLM-L6-v2`). This ensures reproducibility in CI and on developer machines without keys.
- **Vector Store:** FAISS with `IndexFlatIP` on normalized embeddings (cosine similarity) for fast retrieval.
- **RAG Prompt:** A short, deterministic prompt is used (documented exactly in `docs/ARCHITECTURE.md`) so that results are reproducible and auditable.
- **LLM Integration:** LLM usage (OpenAI) is optional; when an LLM is unavailable due to quota or key issues, the pipeline falls back to deterministic behavior (e.g., return top retrieved questions or a simple template-based generator).

## What’s Included
- `src/` — core modules (embedding engine, rag engine, parsers)
- `scripts/` — build, test and example scripts
- `vector_store/` — FAISS indices and metadata
- `docs/` — architecture + PlantUML



## Limitations & Next Steps
- Add unit tests for the RAG parser and LLM parsing logic.
- Add more robust template-based generation for cases where the LLM is unavailable (currently we fall back to retrieved questions).
- Add integration tests with a local LLM (e.g., Llama.cpp or ONNX) for better deterministic generation in air-gapped environments.

## Reproducible Runbook
1. Run environment setup (see README).
2. Run `python scripts/build_interview_index.py` (index building will use local embeddings if OpenAI is not available).
3. Run `python scripts/run_rag_example.py` to test retrieval + generation.






📁 .github/workflows/ci.yml
Purpose: Continuous Integration (CI)

Contains:
  GitHub Actions pipeline
  Runs tests (ci_smoke_test.py)
  Ensures code doesn’t break on commits

Why it exists:
  Industry best practice
  Shows project is production-ready

📁 data/
This folder contains raw and processed datasets.

## data/interview_questions/questions.txt

Contains:
  Plain-text interview questions
  One question per line

Used for:
  Building the FAISS vector index
  Retrieval in RAG

Why TXT:
  Simple
  Fast to load
  Ideal for semantic search


# data/job_descriptions/

Contains:
  Job descriptions in .txt format
  Each file = one job posting

Used for:
  Resume–job matching
  Context for RAG + LLM

# data/resumes/

Contains:
  Candidate resumes in .txt format

Used for:
  Matching
  Query input to RAG
  Personalizing interview questions

# data/skills/

Contains:
  Skill taxonomy files (JSON / TXT)

Used for:
  Skill extraction
  Resume understanding
  Matching improvement

# ci_small_questions.txt

Contains:
  Very small interview-question subset

Used for:
  CI testing
  Fast automated checks

📁 docs/
# ARCHITECTURE.md

Contains:
  Written explanation of system architecture
  Pipeline description
  Design decisions

Used for:
  Academic documentation
  Readability for reviewers

# architecture.puml

Contains:
  UML diagram source (PlantUML)

Used for:
  Visual architecture diagrams
  Documentation / reports

📁 modules/
Low-level utility logic (no ML here).

# modules/utils.py

Contains:
  load_txt_folder()
  File reading helpers

Used for:
  Loading resumes
  Loading jobs
  Loading questions

# modules/matching.py

Contains:
  Resume–job similarity logic
  Usually cosine similarity

Used for:
  Ranking jobs for each resume

📁 scripts/
Offline preprocessing scripts
(run once, not during inference)

# build_interview_index.py

Contains:
  Builds FAISS index from questions.txt
  Stores vectors + metadata

Used for:
  RAG retrieval

# build_job_descriptions.py

Contains:
  Converts raw job dataset → .txt

# build_resumes.py

Contains:
  Converts resume dataset → .txt

# build_skill_list.py

Contains:
  Aggregates skill JSON files
  Creates unified skill list

# check_embeddings.py

Contains:
  Debug script
  Validates embeddings + FAISS index

# ci_smoke_test.py

Contains:
  Minimal pipeline test

Used for:
  CI validation

# question_loader.py

Contains:
  Loads interview question datasets
  Converts to TXT

# run_rag_example.py

Contains:
  Standalone demo of RAG
  Useful for debugging

📁 src/
Core ML / NLP pipeline

# embedding_engine.py

Contains:
  SentenceTransformer model
  FAISS index handling
  embed() and search()

Role:
  Backbone of RAG
  Semanaic similarity engine

# rag_engine.py

Contains:
  RAG logic
  Retrieval + LLM prompt assembly
  Parsing LLM output

Role:
  Connects retrieval with generation

# llm_engine.py

Contains:
  Local LLM interface (Ollama)
  Prompt execution

Role:
  Open-source LLM inference
  No paid APIs

# interview_copilot.py

Contains:
  Orchestrates interview generation
  Calls RAG engine

Role:
  Main “interview assistant” logic

# matcher.py

Contains:
  Wrapper around matching logic

Role:
  Keeps pipeline clean

# resume_parser.py

Contains:
  Resume text cleaning
  Section extraction (optional)

# skill_extractor.py

Contains:
  Extracts skills from resumes
  Matches against skill taxonomy

📁 vectorstore/

Contains:
  FAISS index files
  Metadata (JSON)

Role:
  Persistent vector database for RAG

📁 venv/

Contains:
  Python virtual environment

Role:
 Dependency isolation

📄 app.py
Main entry point

What it does:
  Loads resumes
  Loads job descriptions
  Matches resumes to jobs
  Runs RAG + LLM
  Prints interview questions