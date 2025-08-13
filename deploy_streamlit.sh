#!/bin/bash

set -e

# Force building for AMD64 architecture (for Fargate)
export DOCKER_DEFAULT_PLATFORM=linux/amd64

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
ECR_REPO="cabruca-stg"

echo "Building and deploying Streamlit application..."

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Streamlit Docker image
echo "Building Streamlit Docker image for AMD64..."
docker build -f Dockerfile.streamlit -t cabruca-streamlit .

# Tag and push
echo "Tagging image..."
docker tag cabruca-streamlit:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:streamlit-latest

echo "Pushing to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:streamlit-latest

# Update Streamlit task definition
cat > /tmp/streamlit-task-def.json <<EOF
{
  "family": "cabruca-stg-streamlit",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "streamlit",
      "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:streamlit-latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "STREAMLIT_SERVER_PORT", "value": "8501"},
        {"name": "STREAMLIT_SERVER_ADDRESS", "value": "0.0.0.0"},
        {"name": "API_URL", "value": "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/streamlit",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "streamlit"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8501/ || exit 1"],
        "interval": 30,
        "timeout": 10,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Register task definition
echo "Registering Streamlit task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/streamlit-task-def.json --region $AWS_REGION > /dev/null

# Update service
echo "Updating Streamlit service..."
aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-streamlit \
  --task-definition cabruca-stg-streamlit \
  --force-new-deployment \
  --region $AWS_REGION > /dev/null

echo "Deployment complete! Checking status..."
sleep 15

# Check service status
aws ecs describe-services \
  --cluster cabruca-stg-cluster \
  --services cabruca-stg-streamlit \
  --region $AWS_REGION \
  --query 'services[*].[serviceName,runningCount,desiredCount]' \
  --output table

echo -e "\nStreamlit will be available at:"
echo "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/streamlit"
echo "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/dashboard"