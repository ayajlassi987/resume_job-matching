# Deployment Guide

## Overview

This guide covers deployment options for the Interview Copilot system, including local development, Docker, and production deployments.

## Prerequisites

- Python 3.10 or higher
- pip package manager
- (Optional) Docker and Docker Compose

## Local Development Setup

### 1. Create Virtual Environment

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

### 4. Build FAISS Index

```bash
python scripts/build_interview_index.py \
  --questions-file data/interview_questions/questions.txt \
  --index-dir vector_store/faiss_index \
  --overwrite \
  --verify
```

### 5. Configure Environment Variables

Create a `.env` file:

```env
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=True
LLM_BACKEND=ollama
LLM_MODEL=mistral
OPENAI_API_KEY=your_key_here  # Optional
```

### 6. Run the API

**Option 1: Using Flask CLI (recommended)**

Set the FLASK_APP environment variable and run:

```bash
# Windows PowerShell
$env:FLASK_APP="api.app"
python -m flask run --host=0.0.0.0 --port=5000

# Linux/Mac
export FLASK_APP=api.app
python -m flask run --host=0.0.0.0 --port=5000
```

**Option 2: Run directly (simpler)**

```bash
python api/app.py
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. **Build and start services:**

```bash
docker-compose up -d
```

2. **Check logs:**

```bash
docker-compose logs -f api
```

3. **Stop services:**

```bash
docker-compose down
```

### Using Docker Only

1. **Build image:**

```bash
docker build -t interview-copilot .
```

2. **Run container:**

```bash
docker run -p 5000:5000 \
  -v $(pwd)/vector_store:/app/vector_store \
  -v $(pwd)/data:/app/data \
  -e LLM_BACKEND=ollama \
  interview-copilot
```

## Production Deployment

### Environment Configuration

For production, set the following environment variables:

```env
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False
LLM_BACKEND=ollama
LLM_MODEL=mistral
LOG_LEVEL=INFO
```

### Using Gunicorn

For production, use Gunicorn as the WSGI server:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api.app:app
```

### Using Systemd (Linux)

Create `/etc/systemd/system/interview-copilot.service`:

```ini
[Unit]
Description=Interview Copilot API
After=network.target

[Service]
User=www-data
WorkingDirectory=/opt/interview-copilot
Environment="PATH=/opt/interview-copilot/.venv/bin"
ExecStart=/opt/interview-copilot/.venv/bin/gunicorn -w 4 -b 0.0.0.0:5000 api.app:app

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable interview-copilot
sudo systemctl start interview-copilot
```

### Using Nginx as Reverse Proxy

Example Nginx configuration:

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## LLM Backend Setup

### Ollama (Recommended for Local)

1. **Install Ollama:**

Visit https://ollama.ai and install for your platform.

2. **Pull model:**

```bash
ollama pull mistral
```

3. **Configure:**

Set `LLM_BACKEND=ollama` and `LLM_MODEL=mistral` in `.env`.

### Hugging Face Transformers

1. **Install dependencies:**

```bash
pip install transformers torch
```

2. **Configure:**

Set `LLM_BACKEND=huggingface` and `LLM_MODEL=model-name` in `.env`.

### vLLM (For GPU Servers)

1. **Install:**

```bash
pip install vllm
```

2. **Configure:**

Set `LLM_BACKEND=vllm` and `LLM_MODEL=model-name` in `.env`.

## Monitoring

### Health Checks

The API provides a health endpoint:

```bash
curl http://localhost:5000/api/health
```

### Monitoring Script

Run the monitoring script:

```bash
python scripts/monitor.py
```

### Logs

Logs are written to:
- Console (stdout)
- File: `logs/app.log` (if configured)

## Troubleshooting

### Common Issues

1. **FAISS index not found:**
   - Run `scripts/build_interview_index.py` to build the index

2. **LLM not responding:**
   - Check that Ollama is running: `ollama list`
   - Verify model is pulled: `ollama pull mistral`

3. **Import errors:**
   - Ensure virtual environment is activated
   - Run `pip install -r requirements.txt`

## Scaling

### Horizontal Scaling

Use a load balancer (e.g., Nginx) with multiple Gunicorn workers:

```bash
gunicorn -w 8 -b 0.0.0.0:5000 api.app:app
```

### Vertical Scaling

- Increase Gunicorn workers based on CPU cores
- Use GPU for LLM inference (vLLM backend)
- Use managed vector database (Pinecone, Weaviate) for large-scale deployments

## Security Considerations

1. **API Authentication:** Add API key authentication for production
2. **Rate Limiting:** Configure appropriate rate limits
3. **Input Validation:** Already handled by Pydantic models
4. **HTTPS:** Use reverse proxy (Nginx) with SSL certificates
5. **Secrets Management:** Use environment variables or secret management services

## Backup and Recovery

### Backup

- FAISS index: `vector_store/faiss_index/`
- Configuration: `.env` file
- Logs: `logs/` directory

### Recovery

1. Restore FAISS index files
2. Restore `.env` configuration
3. Restart the service

