FROM python:3.12-slim

WORKDIR /app

# Python & pip configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Copy dependency file first to maximise Docker layer caching
COPY requirements.txt .

# Install all dependencies (CPU-only PyTorch)
RUN python -m pip install \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# Copy application source
COPY . .

# Create a non-root user and writable persistent-data mount points
RUN addgroup --system app && \
    adduser \
    --system \
    --ingroup app \
    --home /home/app \
    app && \
    mkdir -p \
    /home/app/.cache/huggingface \
    /app/data/uploads && \
    chown -R app:app /home/app /app

# Configure Hugging Face cache
ENV HOME=/home/app \
    HF_HOME=/home/app/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/app/.cache/huggingface

# Run as non-root
USER app

# Application port
EXPOSE 8000

# Container health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
