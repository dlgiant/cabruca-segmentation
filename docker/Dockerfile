# Dockerfile for Cabruca Segmentation API
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY api_server.py .

# Create directories for uploads and results
RUN mkdir -p api_uploads api_results outputs

# Expose port
EXPOSE 8000

# Set environment variables
ENV MODEL_PATH=/app/outputs/checkpoint_best.pth
ENV PLANTATION_DATA_PATH=/app/plantation-data.json
ENV PYTHONPATH=/app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the API server
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]