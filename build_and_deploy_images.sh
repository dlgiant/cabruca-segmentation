#!/bin/bash

# Build and deploy Docker images to ECR for ECS services

set -e

# Configuration
AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
ECR_REPO_API="cabruca-stg"
IMAGE_TAG="latest"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Create ECR repository if it doesn't exist
echo "Ensuring ECR repository exists..."
aws ecr describe-repositories --repository-names $ECR_REPO_API --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO_API --region $AWS_REGION

# Build API image
echo "Building API Docker image..."
docker build -f docker/Dockerfile -t cabruca-api .

# Tag API image for ECR
echo "Tagging API image..."
docker tag cabruca-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:api-latest
docker tag cabruca-api:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:api-$IMAGE_TAG

# Build Streamlit image
echo "Building Streamlit Docker image..."
cat > Dockerfile.streamlit <<EOF
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Streamlit and dependencies
RUN pip install --no-cache-dir \
    streamlit \
    pandas \
    numpy \
    plotly \
    boto3 \
    Pillow

# Copy Streamlit app
COPY streamlit_app.py .
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p outputs data api_uploads api_results

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
EOF

docker build -f Dockerfile.streamlit -t cabruca-streamlit .

# Tag Streamlit image for ECR
echo "Tagging Streamlit image..."
docker tag cabruca-streamlit:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:streamlit-latest
docker tag cabruca-streamlit:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:streamlit-$IMAGE_TAG

# Build Annotation image (reuse streamlit for now)
echo "Tagging Annotation image..."
docker tag cabruca-streamlit:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:annotation-latest

# Push images to ECR
echo "Pushing images to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:api-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:api-$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:streamlit-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:streamlit-$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO_API:annotation-latest

echo "Docker images built and pushed successfully!"

# Force new deployments on ECS services
echo "Forcing new deployments on ECS services..."
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-api --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-streamlit --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-annotation --force-new-deployment --region $AWS_REGION

echo "ECS services updated. Waiting for deployments to complete..."

# Wait for services to stabilize
echo "Waiting for API service to stabilize..."
aws ecs wait services-stable --cluster cabruca-stg-cluster --services cabruca-stg-api --region $AWS_REGION || true

echo "Deployment complete! Check service status:"
aws ecs describe-services --cluster cabruca-stg-cluster --services cabruca-stg-api cabruca-stg-streamlit --region $AWS_REGION --query 'services[*].[serviceName,runningCount,desiredCount]' --output table