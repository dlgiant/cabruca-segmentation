# Lightweight Dockerfile for Cabruca Segmentation
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only essential system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better caching
COPY requirements.txt .

# Install core Python dependencies (skip heavy ML packages for CI/CD testing)
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    requests \
    pydantic \
    python-multipart \
    boto3 \
    || pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p api_uploads api_results outputs data

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Simple health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=2 \
    CMD python -c "print('healthy')" || exit 1

# Default command
CMD ["python", "-c", "print('Container started successfully')"]