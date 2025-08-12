# 🚀 Cabruca Segmentation Deployment Guide

## Quick Start - Local Deployment ✅

Successfully deployed the Cabruca Segmentation application with both API and Dashboard!

### Prerequisites Installed
- ✅ Python 3.9.6
- ✅ Virtual environment (venv)
- ✅ All dependencies installed

### What's Running

#### 1. FastAPI Server
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Health**: http://localhost:8000/health
- **Features**:
  - Image segmentation endpoint
  - Health checks
  - Metrics tracking
  - AgentOps integration
  - Mock mode for testing (no ML model required)

#### 2. Streamlit Dashboard
- **URL**: http://localhost:8501
- **Features**:
  - Image upload and segmentation
  - Real-time analytics
  - Plantation mapping
  - AWS integration status
  - Interactive visualizations

## Installation Steps

### 1. Clone Repository
```bash
git clone https://github.com/dlgiant/cabruca-segmentation.git
cd cabruca-segmentation
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install --upgrade pip
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install fastapi "uvicorn[standard]" python-multipart
pip install streamlit plotly agentops
pip install opencv-python Pillow numpy scipy pandas
```

### 4. Setup Directories
```bash
mkdir -p outputs api_uploads api_results
```

### 5. Start Services

#### Option A: Simple API (Recommended for Testing)
```bash
# Terminal 1 - Start API
source venv/bin/activate
python simple_api.py

# Terminal 2 - Start Dashboard
source venv/bin/activate
streamlit run streamlit_app.py
```

#### Option B: Full API Server
```bash
# Terminal 1 - Start API
source venv/bin/activate
python api_server.py --model outputs/checkpoint_best.pth

# Terminal 2 - Start Dashboard
source venv/bin/activate
streamlit run streamlit_app.py
```

## Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API Server | http://localhost:8000 | REST API endpoints |
| API Docs | http://localhost:8000/docs | Interactive API documentation |
| Dashboard | http://localhost:8501 | Streamlit web interface |
| Health Check | http://localhost:8000/health | API health status |

## Testing the Deployment

### 1. Test API Health
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "timestamp": "2025-08-12T14:55:53.249803",
  "model_loaded": false,
  "version": "1.0.0"
}
```

### 2. Test Image Segmentation
```bash
# Upload a test image
curl -X POST "http://localhost:8000/segment" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@test_image.jpg"
```

### 3. Access Dashboard
Open browser to http://localhost:8501 and:
1. Upload an image in the "Image Segmentation" tab
2. Click "Run Segmentation"
3. View results and analytics

## Docker Deployment (Optional)

### Build Docker Image
```bash
cd docker
docker-compose build
```

### Run with Docker Compose
```bash
docker-compose up -d
```

### Access Services
- API: http://localhost:8000
- Dashboard: http://localhost:8501
- Nginx Proxy: http://localhost:80

## Cloud Deployment Options

### AWS ECS (Production)
1. Build and push Docker image to ECR
2. Update ECS task definition
3. Deploy ECS service with ALB

### AWS Lambda (Serverless)
1. Package API as Lambda function
2. Upload model to S3
3. Configure API Gateway

### EC2 Instance (Simple)
1. Launch EC2 with Deep Learning AMI
2. Clone repository
3. Install dependencies
4. Run with systemd service

## Environment Variables

Create a `.env` file for configuration:

```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=outputs/checkpoint_best.pth
PLANTATION_DATA_PATH=plantation-data.json

# AWS Configuration
AWS_REGION=sa-east-1
S3_BUCKET=cabruca-mvp-mvp-agent-artifacts-919014037196
DYNAMODB_TABLE=cabruca-mvp-mvp-agent-state

# AgentOps
AGENTOPS_API_KEY=89428585-c28a-419b-87fe-6ce52d6c47e5
```

## Monitoring

### AgentOps Dashboard
- Visit: https://app.agentops.ai
- Track API calls and segmentation requests
- Monitor performance metrics

### AWS CloudWatch
- Dashboard: https://console.aws.amazon.com/cloudwatch/
- Monitor Lambda functions and infrastructure

## Troubleshooting

### Common Issues

1. **Port Already in Use**
```bash
# Kill process on port 8000
lsof -ti:8000 | xargs kill -9

# Kill process on port 8501
lsof -ti:8501 | xargs kill -9
```

2. **Module Import Errors**
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

3. **Model Not Found**
- The simple_api.py runs in mock mode without a model
- To use real model, train one or download pre-trained weights

## Next Steps

1. **Train ML Model**
   - Prepare dataset with annotation tools
   - Train model using `train.py`
   - Save checkpoint to `outputs/`

2. **Configure AWS Integration**
   - Set up S3 buckets for model storage
   - Configure DynamoDB for state management
   - Connect Lambda functions

3. **Production Deployment**
   - Set up SSL certificates
   - Configure domain name
   - Enable auto-scaling
   - Set up monitoring alerts

## Support

- GitHub Issues: https://github.com/dlgiant/cabruca-segmentation/issues
- API Documentation: http://localhost:8000/docs
- AgentOps Support: https://docs.agentops.ai

---

**Deployment Status: ✅ Successfully Deployed Locally**

The Cabruca Segmentation System is now running and ready for use!