# Architecture Overview ✅

This document explains the high-level architecture of the AI-Resume-Interview-Copilot project and the RAG prompt used for tailored interview question generation.

## Components

- **app.py** (Flask): exposes endpoints to match resumes and to start interview sessions.
- **src/**: core modules:
  - `resume_parser.py` — cleans and normalizes resume text
  - `skill_extractor.py` — extracts skills from text via keyword matching
  - `embedding_engine.py` — provides embeddings (OpenAI preferred, fallback to Sentence-Transformers), builds and searches FAISS index
  - `matcher.py` — matches resumes against job descriptions using embeddings
  - `rag_engine.py` — retrieves relevant interview questions from FAISS and uses the LLM to generate tailored questions (RAG)
  - `llm_engine.py` — wrapper for calling LLMs (OpenAI) with graceful fallback
  - `interview_copilot.py` — top-level orchestration for interactive interview sessions and answers evaluation
- **vector_store/faiss_index/** — persisted FAISS index files: `index.faiss`, `embeddings.npy`, `meta.json`
- **data/** — source data: resumes, job descriptions, interview questions

## Data & Flow

1. Resume text → `resume_parser` → cleaned text
2. Candidate skills extracted via `skill_extractor`
3. `rag_engine` queries the FAISS index (built by `embedding_engine`) with the resume and job description to retrieve top-N relevant interview questions
4. The retrieved questions + resume + job description are sent to the LLM via `llm_engine` using the RAG prompt to generate 5 tailored interview questions
5. The `interview_copilot` orchestrates this flow and can evaluate answers using embeddings & matching

## RAG Prompt (exact) 💡

You MUST use the following prompt when generating tailored interview questions (kept verbatim):

"You are a professional interviewer.

Candidate Resume:
{resume}

Job Description:
{job}

Relevant Interview Questions:
{retrieved_questions}

Generate 5 tailored interview questions."

This prompt is intentionally concise and deterministic to help with reproducible results.

## Diagram

The PlantUML source is included at `docs/architecture.puml`. Render it via PlantUML to generate PNG/SVG, or view it in your editor extension.

> **Tip:** The project includes local fallbacks to make the RAG/retrieval pipeline deterministic for CI (no external API key needed). See the README for how to run tests locally and in CI.