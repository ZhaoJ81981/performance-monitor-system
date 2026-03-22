# Performance Monitor System - ML Pipeline Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install smartmontools for disk SMART data (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
    smartmontools \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CPU-only (adjust for GPU if needed)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install TensorFlow CPU
RUN pip install --no-cache-dir tensorflow-cpu

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/logs /app/models /app/data /app/config

# Create non-root user
RUN useradd -m -u 1000 pmsuser && \
    chown -R pmsuser:pmsuser /app
USER pmsuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)" || exit 1

# Default command (can be overridden)
CMD ["python", "src/ml_pipeline/train.py", "--config", "config/local.yaml"]

# Expose metrics port (optional)
EXPOSE 8080

# Labels
LABEL maintainer="Performance Monitor System Team"
LABEL version="1.0.0"
LABEL description="ML Pipeline for Performance Monitor System"

# Build arguments
ARG BUILD_DATE
ARG VCS_REF
LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.vcs-url="https://github.com/ZhaoJ81981/performance-monitor-system"