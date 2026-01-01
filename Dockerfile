# SparsePlug Production Deployment
# Full UPG-PAC V2 Platform with Adaptive Server
# For Render.com, Railway.app, or Fly.io

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    libgomp1 \
    build-essential \
    git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire application
COPY . .

# Create necessary directories
RUN mkdir -p models logs uploads downloads

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Expose port (dynamic for cloud platforms)
EXPOSE ${PORT}

# Health check
HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Start adaptive server
CMD uvicorn adaptive_server:app --host 0.0.0.0 --port ${PORT} --workers 1
# Cache bust: 1767287435
