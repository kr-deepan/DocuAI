# DocuAI Production Deployment Guide

This guide covers all deployment strategies for **DocuAI: Contradiction-Aware Multi-Agent Document Intelligence System**, ranging from 100% free one-click cloud hosting to containerized multi-service orchestration.

---

## Table of Contents
1. [Option 1: Streamlit Community Cloud (Recommended for Resumes & Demos)](#1-option-1-streamlit-community-cloud-recommended-for-resumes--demos)
2. [Option 2: Docker Single-Container Deployment](#2-option-2-docker-single-container-deployment)
3. [Option 3: Docker Compose (Decoupled Full-Stack: FastAPI + Streamlit)](#3-option-3-docker-compose-decoupled-full-stack-fastapi--streamlit)
4. [Option 4: Hugging Face Spaces](#4-option-4-hugging-face-spaces)
5. [Environment Variables & Secrets Management](#5-environment-variables--secrets-management)
6. [Cloud-Specific Production Considerations](#6-cloud-specific-production-considerations)

---

## 1. Option 1: Streamlit Community Cloud (Recommended for Resumes & Demos)

Streamlit Community Cloud is 100% free, connects directly to your GitHub repository, provides an automatic HTTPS certificate, and yields a permanent URL ideal for technical portfolios.

### Step-by-Step Deployment:
1. Push your latest code to GitHub:
   ```bash
   git push origin main
   ```
2. Navigate to **[share.streamlit.io](https://share.streamlit.io)** and log in with your GitHub account.
3. Click **"New app"** (or **"Create app"**).
4. Configure the deployment settings:
   - **Repository**: `kr-deepan/DocuAI` (or your GitHub fork/repo)
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: Choose a custom subdomain (e.g. `docuai.streamlit.app`)
5. **Configure Secrets**:
   - Click **"Advanced settings"** (or go to App Settings > **Secrets** after creating).
   - Enter your OpenRouter API key in TOML format:
     ```toml
     OPENROUTER_API_KEY = "your-openrouter-api-key-here"
     ```
   - *(Note: If you leave this empty, DocuAI will automatically operate in offline mock mode so visitors can test the UI without errors).*
6. Click **"Deploy!"**.

### How it Works Behind the Scenes:
- **System Binaries**: Streamlit Cloud detects `packages.txt` and automatically installs `tesseract-ocr` and `libtesseract-dev`.
- **Python Dependencies**: Installs packages listed in `requirements.txt`.
- **Auto-Ingestion on Clean Launch**: When ChromaDB is empty on first boot, `src/ui/app.py` automatically detects 0 indexed chunks and triggers `IngestionPipeline().run()`, indexing the sample legal corpus (`data/sample_docs/`) into ChromaDB and building the BM25 index in $< 10\text{ seconds}$.

---

## 2. Option 2: Docker Single-Container Deployment

DocuAI includes a multi-stage production `Dockerfile` that packages Python 3.10, system Tesseract OCR dependencies, and the full multi-agent application.

### Build and Run:
```bash
# Build the Docker image
docker build -t docuai:latest .

# Run the container with your API key
docker run -d \
  --name docuai-app \
  -p 8501:8501 \
  -e OPENROUTER_API_KEY="your-openrouter-api-key-here" \
  docuai:latest
```

Access the Streamlit Dashboard at:
- **`http://localhost:8501`**

### Inspecting Container Health:
```bash
docker ps
docker logs -f docuai-app
```

---

## 3. Option 3: Docker Compose (Decoupled Full-Stack: FastAPI + Streamlit)

For production enterprise setups, DocuAI supports multi-service orchestration via `docker-compose.yml`, running both the **asynchronous FastAPI REST server** and the **interactive Streamlit frontend** with shared persistent storage.

### Services:
- `docuai-ui`: Streamlit frontend on port `8501`
- `docuai-api`: FastAPI REST backend on port `8000`

### Launching:
```bash
# Set your API key in environment or .env file
export OPENROUTER_API_KEY="your-api-key"

# Launch all services
docker compose up -d --build
```

### Endpoints:
- **Streamlit Trace UI**: `http://localhost:8501`
- **FastAPI REST API**: `http://localhost:8000`
- **FastAPI Swagger Docs**: `http://localhost:8000/docs`
- **Health Check**: `http://localhost:8000/health`
- **Budget Metrics**: `http://localhost:8000/api/budget`

### Stopping:
```bash
docker compose down
```

---

## 4. Option 4: Hugging Face Spaces

Hugging Face Spaces is an alternative free hosting provider popular within the AI engineering community.

### Deployment Steps:
1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces).
2. Choose **Streamlit** (or **Docker**) as the Space SDK.
3. Push the DocuAI repository to your Hugging Face Space repository:
   ```bash
   git remote add space https://huggingface.co/spaces/YOUR_USERNAME/DocuAI
   git push space main
   ```
4. In the Space settings, add your `OPENROUTER_API_KEY` under **Repository secrets**.

---

## 5. Environment Variables & Secrets Management

DocuAI reads configuration from `.env`, system environment variables, and `st.secrets`:

| Variable | Description | Default |
|---|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API authentication token | `""` (runs in offline mock mode if empty) |
| `OPENROUTER_BASE_URL` | OpenRouter API endpoint URL | `https://openrouter.ai/api/v1` |
| `ORCHESTRATOR_MODEL` | Fast model for query intent and decomposition | `poolside/laguna-xs-2.1:free` |
| `RETRIEVAL_MODEL` | Model for retrieval and query generation | `poolside/laguna-xs-2.1:free` |
| `CRITIC_MODEL` | Model for grounding & verification checks | `poolside/laguna-s-2.1:free` |
| `SYNTHESIS_MODEL` | High-capability model for answer synthesis | `nvidia/nemotron-3-ultra-550b-a55b:free` |
| `EMBEDDING_MODEL` | Dense vector embedding model | `nvidia/llama-nemotron-embed-vl-1b-v2:free` |
| `DAILY_REQUEST_BUDGET`| Hard request ceiling for pooled quota guard | `45` |
| `CHROMA_PERSIST_DIR` | Local disk directory for ChromaDB storage | `./data/chroma_db` |

---

## 6. Cloud-Specific Production Considerations

### Ephemeral Storage & Re-Indexing
Cloud platforms (such as Streamlit Community Cloud or serverless containers) deploy with ephemeral disk storage.
- DocuAI's `src/ui/app.py` checks `vector_store.get_count()` on startup.
- If chunks are missing, it triggers an instant in-process ingestion pass over `data/sample_docs/`.
- Users can also click the **"🔄 Re-Index Corpus"** button in the sidebar to re-process all documents at any time.

### Daily Request Quota Guard
- OpenRouter free-tier accounts share a pooled 50-request daily limit.
- DocuAI enforces a strict **45-request/day ceiling** (`src/core/budget.py`) with an SQLite response cache (`src/core/cache.py`) that eliminates quota consumption on repeated test queries.
- On cloud platforms, cached responses persist in the container's SQLite database to maintain zero-cost demo reliability.
