# =============================================================================
# ISRO Satellite Image Classification — Dockerfile
# Multi-stage build for minimal production image
# =============================================================================

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git libgdal-dev gdal-bin \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy only installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# System runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgdal-dev libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Create directories
RUN mkdir -p checkpoints logs results data

# Non-root user for security
RUN useradd -m -u 1000 satuser && chown -R satuser:satuser /app
USER satuser

# Environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TORCH_HOME=/app/.cache/torch

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default port
EXPOSE 8000 8501

# Default command — FastAPI server
CMD ["uvicorn", "src.deployment.api:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
