# Deployment Guide for Cabruca Segmentation System

This guide covers various deployment options for the Cabruca Segmentation system, from local development to production cloud deployments.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Deployment Options](#deployment-options)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring and Maintenance](#monitoring-and-maintenance)

## Quick Start

### Local Development Server

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start API server
python api_server.py --model outputs/checkpoint_best.pth

# 3. Start Streamlit dashboard (in another terminal)
streamlit run src/inference/interactive_viewer.py

# Access:
# API: http://localhost:8000
# Dashboard: http://localhost:8501
```

## Deployment Options

### 1. FastAPI REST API

**Best for:** Production APIs, microservices, integration with existing systems

```python
# api_server.py
python api_server.py \
    --host 0.0.0.0 \
    --port 8000 \
    --model outputs/checkpoint_best.pth \
    --workers 4
```

**Features:**
- Async request handling
- Auto-generated documentation at `/docs`
- Batch processing support
- Job queue management
- WebSocket support for real-time updates

**Configuration:**
```yaml
# config/api_config.yaml
server:
  host: 0.0.0.0
  port: 8000
  workers: 4
  timeout: 300
  max_upload_size: 100MB

model:
  path: outputs/checkpoint_best.pth
  device: auto
  batch_size: 8
  
cache:
  enabled: true
  ttl: 3600
  max_size: 1000
```

### 2. Streamlit Application

**Best for:** Interactive demos, data exploration, internal tools

```bash
# Basic launch
streamlit run src/inference/interactive_viewer.py

# Production launch with custom config
streamlit run src/inference/interactive_viewer.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --server.maxUploadSize 100
```

**Streamlit Configuration:**
```toml
# .streamlit/config.toml
[server]
port = 8501
address = "0.0.0.0"
headless = true
maxUploadSize = 100
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#4CAF50"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
```

### 3. Flask Lightweight API

**Best for:** Simple deployments, embedded systems, edge devices

```python
# flask_app.py
from flask import Flask, request, jsonify
import torch
from src.inference.batch_inference import BatchInferenceEngine

app = Flask(__name__)
engine = BatchInferenceEngine("outputs/checkpoint_best.pth")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    result = engine.process_single(file)
    return jsonify(result.to_dict())

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 4. Gradio Interface

**Best for:** Quick demos, shareable links, minimal setup

```python
# gradio_app.py
import gradio as gr
from src.inference.batch_inference import BatchInferenceEngine

engine = BatchInferenceEngine("outputs/checkpoint_best.pth")

def process_image(image):
    result = engine.process_single(image)
    visualization = engine.create_visualization(image, result)
    metrics = {
        "Trees Detected": len(result.trees),
        "Canopy Density": f"{result.canopy_density:.2%}",
        "Cacao Trees": sum(1 for t in result.trees if t.species == 'cacao'),
        "Shade Trees": sum(1 for t in result.trees if t.species == 'shade')
    }
    return visualization, metrics

interface = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath"),
    outputs=[
        gr.Image(label="Segmentation Result"),
        gr.JSON(label="Metrics")
    ],
    title="Cabruca Segmentation",
    description="Upload an image to analyze cabruca plantation"
)

interface.launch(share=True)
```

## Docker Deployment

### Single Container

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY api_server.py .
COPY outputs/checkpoint_best.pth ./outputs/

# Expose ports
EXPOSE 8000

# Run API server
CMD ["python", "api_server.py", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
# Build image
docker build -t cabruca-segmentation .

# Run container
docker run -d \
    --name cabruca-api \
    -p 8000:8000 \
    -v $(pwd)/data:/app/data \
    cabruca-segmentation
```

### Docker Compose Multi-Service

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./outputs:/app/outputs
      - ./data:/app/data
    environment:
      - MODEL_PATH=/app/outputs/checkpoint_best.pth
      - DEVICE=cpu
    restart: unless-stopped
    
  streamlit:
    build: .
    command: streamlit run src/inference/interactive_viewer.py
    ports:
      - "8501:8501"
    volumes:
      - ./outputs:/app/outputs
      - ./data:/app/data
    depends_on:
      - api
    restart: unless-stopped
    
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - streamlit
    restart: unless-stopped

  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    command: redis-server --maxmemory 256mb --maxmemory-policy allkeys-lru
    restart: unless-stopped
```

Launch all services:
```bash
docker-compose up -d
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cabruca-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: cabruca-api
  template:
    metadata:
      labels:
        app: cabruca-api
    spec:
      containers:
      - name: api
        image: cabruca-segmentation:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        env:
        - name: MODEL_PATH
          value: "/app/outputs/checkpoint_best.pth"
        volumeMounts:
        - name: model-storage
          mountPath: /app/outputs
      volumes:
      - name: model-storage
        persistentVolumeClaim:
          claimName: model-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: cabruca-api-service
spec:
  selector:
    app: cabruca-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

Deploy to Kubernetes:
```bash
kubectl apply -f k8s-deployment.yaml
kubectl get pods
kubectl get services
```

## Cloud Deployment

### AWS EC2

```bash
# 1. Launch EC2 instance (Ubuntu 20.04, t3.large or larger)

# 2. SSH into instance
ssh -i your-key.pem ubuntu@ec2-instance-ip

# 3. Install dependencies
sudo apt update
sudo apt install -y python3-pip nginx git
git clone https://github.com/yourrepo/cabruca-segmentation.git
cd cabruca-segmentation
pip3 install -r requirements.txt

# 4. Configure systemd service
sudo nano /etc/systemd/system/cabruca.service
```

```ini
[Unit]
Description=Cabruca Segmentation API
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/cabruca-segmentation
ExecStart=/usr/bin/python3 api_server.py --host 0.0.0.0 --port 8000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
# 5. Start service
sudo systemctl enable cabruca
sudo systemctl start cabruca

# 6. Configure Nginx
sudo nano /etc/nginx/sites-available/cabruca
```

### Google Cloud Platform

```bash
# 1. Build container
gcloud builds submit --tag gcr.io/PROJECT_ID/cabruca-segmentation

# 2. Deploy to Cloud Run
gcloud run deploy cabruca-api \
    --image gcr.io/PROJECT_ID/cabruca-segmentation \
    --platform managed \
    --region us-central1 \
    --memory 4Gi \
    --cpu 2 \
    --timeout 300 \
    --concurrency 10 \
    --port 8000
```

### Azure Container Instances

```bash
# 1. Build and push to Azure Container Registry
az acr build --registry myregistry --image cabruca-segmentation .

# 2. Deploy container instance
az container create \
    --resource-group myResourceGroup \
    --name cabruca-api \
    --image myregistry.azurecr.io/cabruca-segmentation:latest \
    --cpu 2 \
    --memory 4 \
    --ports 8000 \
    --dns-name-label cabruca-api
```

### Heroku

```yaml
# heroku.yml
build:
  docker:
    web: Dockerfile
run:
  web: python api_server.py --host 0.0.0.0 --port $PORT
```

```bash
# Deploy
heroku create cabruca-segmentation
heroku stack:set container
git push heroku main
```

## Performance Optimization

### 1. Model Optimization

```python
# Quantization
import torch.quantization as quantization

quantized_model = quantization.quantize_dynamic(
    model, {nn.Linear, nn.Conv2d}, dtype=torch.qint8
)

# TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# ONNX Export
torch.onnx.export(
    model, dummy_input, "model.onnx",
    opset_version=11,
    do_constant_folding=True
)
```

### 2. Caching Strategy

```python
# Redis caching
import redis
import pickle

redis_client = redis.Redis(host='localhost', port=6379, db=0)

def cached_inference(image_hash):
    # Check cache
    cached = redis_client.get(image_hash)
    if cached:
        return pickle.loads(cached)
    
    # Compute result
    result = engine.process_single(image_path)
    
    # Cache result
    redis_client.setex(
        image_hash, 
        3600,  # 1 hour TTL
        pickle.dumps(result)
    )
    return result
```

### 3. Load Balancing

```nginx
# nginx.conf
upstream cabruca_api {
    least_conn;
    server api1:8000 weight=3;
    server api2:8000 weight=2;
    server api3:8000 weight=1;
}

server {
    listen 80;
    location / {
        proxy_pass http://cabruca_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 4. Async Processing

```python
# Celery configuration
from celery import Celery

celery_app = Celery('cabruca', broker='redis://localhost:6379')

@celery_app.task
def process_image_async(image_path):
    result = engine.process_single(image_path)
    return result.to_dict()

# API endpoint
@app.post("/process-async")
async def process_async(file: UploadFile):
    task = process_image_async.delay(file.filename)
    return {"task_id": task.id}
```

## Monitoring and Maintenance

### 1. Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": engine is not None,
        "gpu_available": torch.cuda.is_available(),
        "timestamp": datetime.now().isoformat()
    }
```

### 2. Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

inference_counter = Counter('inference_total', 'Total inferences')
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')

@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type="text/plain")
```

### 3. Logging

```python
import logging
from logging.handlers import RotatingFileHandler

# Configure logging
logging.basicConfig(
    handlers=[
        RotatingFileHandler('api.log', maxBytes=10485760, backupCount=5),
        logging.StreamHandler()
    ],
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
```

### 4. Automated Backups

```bash
#!/bin/bash
# backup.sh
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"

# Backup model
cp outputs/checkpoint_best.pth $BACKUP_DIR/model_$TIMESTAMP.pth

# Backup database
pg_dump cabruca_db > $BACKUP_DIR/db_$TIMESTAMP.sql

# Upload to S3
aws s3 cp $BACKUP_DIR s3://cabruca-backups/ --recursive

# Clean old backups (keep last 30 days)
find $BACKUP_DIR -mtime +30 -delete
```

### 5. Auto-scaling

```yaml
# k8s-autoscale.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: cabruca-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: cabruca-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

## Security Considerations

### 1. API Authentication

```python
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/secure-inference")
async def secure_inference(
    file: UploadFile,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    # Verify token
    if not verify_token(credentials.credentials):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Process request
    return await inference(file)
```

### 2. Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/inference")
@limiter.limit("10/minute")
async def rate_limited_inference(request: Request, file: UploadFile):
    return await inference(file)
```

### 3. Input Validation

```python
def validate_image(file: UploadFile):
    # Check file size
    if file.size > 100 * 1024 * 1024:  # 100MB
        raise HTTPException(status_code=413, detail="File too large")
    
    # Check file type
    allowed_types = ['image/jpeg', 'image/png', 'image/tiff']
    if file.content_type not in allowed_types:
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    # Validate image
    try:
        img = Image.open(file.file)
        img.verify()
    except:
        raise HTTPException(status_code=400, detail="Invalid image file")
```

## Troubleshooting

### Common Issues

1. **Out of Memory**
   ```bash
   # Reduce batch size
   python api_server.py --batch-size 1
   
   # Use CPU instead of GPU
   python api_server.py --device cpu
   ```

2. **Slow Inference**
   ```python
   # Enable model optimization
   model = torch.jit.script(model)
   model = torch.quantization.quantize_dynamic(model)
   ```

3. **Connection Timeouts**
   ```nginx
   # Increase timeout in nginx
   proxy_read_timeout 300s;
   proxy_connect_timeout 30s;
   ```

4. **CORS Issues**
   ```python
   from fastapi.middleware.cors import CORSMiddleware
   
   app.add_middleware(
       CORSMiddleware,
       allow_origins=["*"],
       allow_methods=["*"],
       allow_headers=["*"]
   )
   ```

## Best Practices

1. **Use environment variables** for configuration
2. **Implement graceful shutdown** handling
3. **Monitor resource usage** continuously
4. **Keep models versioned** and backed up
5. **Use health checks** for automatic recovery
6. **Implement request tracing** for debugging
7. **Set up alerts** for critical metrics
8. **Document API changes** thoroughly
9. **Test deployments** in staging first
10. **Plan for disaster recovery**

## Support

For deployment assistance:
- GitHub Issues: [Report problems](https://github.com/yourrepo/issues)
- Documentation: [Full docs](https://docs.cabruca.ai)
- Community: [Discord server](https://discord.gg/cabruca)