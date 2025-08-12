# Multi-stage Dockerfile for Cabruca Segmentation
# Build stage
FROM python:3.11-slim as builder

# Set working directory
WORKDIR /app

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Runtime stage
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY src/ ./src/
COPY api_server.py .
COPY *.py ./

# Copy configuration files if they exist
COPY *.json ./
COPY *.yaml ./
COPY *.yml ./

# Copy terraform directory for deployment scripts
COPY terraform/ ./terraform/

# Create directories for uploads, results, and outputs
RUN mkdir -p api_uploads api_results outputs data/processed data/raw

# Expose port for API
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/outputs/checkpoint_best.pth
ENV PLANTATION_DATA_PATH=/app/plantation-data.json
ENV PYTHONPATH=/app:/root/.local/lib/python3.11/site-packages
ENV PATH=/root/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Default command - can be overridden
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]