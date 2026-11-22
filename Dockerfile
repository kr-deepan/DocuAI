# Multi-stage production Dockerfile for DocuAI
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including Tesseract OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source and sample data
COPY src/ ./src/
COPY data/ ./data/
COPY docs/ ./docs/
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/
COPY app.py .
COPY README.md .

# Create necessary runtime directories
RUN mkdir -p data/chroma_db logs

# Expose ports: 8501 for Streamlit, 8000 for FastAPI
EXPOSE 8501 8000

# Healthcheck for container stability
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: launch Streamlit UI
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
