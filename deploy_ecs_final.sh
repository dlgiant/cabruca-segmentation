#!/bin/bash

# Final ECS deployment with working Python server

set -e

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
SOURCE_IMAGE="919014037196.dkr.ecr.sa-east-1.amazonaws.com/cabruca-segmentation:latest"

echo "Final ECS deployment with working configuration..."

# API task definition - runs simple_health_server.py
cat > /tmp/api-task-def.json <<EOF
{
  "family": "cabruca-stg-api",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "$SOURCE_IMAGE",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PYTHONUNBUFFERED", "value": "1"}
      ],
      "command": ["python", "simple_health_server.py"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/api",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "curl -f http://localhost:8000/health || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Streamlit task definition - runs on port 8501
cat > /tmp/streamlit-task-def.json <<EOF
{
  "family": "cabruca-stg-streamlit",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "256",
  "memory": "512",
  "executionRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "streamlit",
      "image": "$SOURCE_IMAGE",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PYTHONUNBUFFERED", "value": "1"}
      ],
      "command": ["python", "-m", "http.server", "8501"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/streamlit",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "streamlit"
        }
      }
    }
  ]
}
EOF

# First, check if simple_health_server.py exists in the image
echo "Checking if simple_health_server.py exists in container..."

# Register new task definitions
echo "Registering API task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/api-task-def.json --region $AWS_REGION --output text > /dev/null

echo "Registering Streamlit task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/streamlit-task-def.json --region $AWS_REGION --output text > /dev/null

# Update services
echo "Updating ECS services..."
aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-api \
  --task-definition cabruca-stg-api \
  --desired-count 1 \
  --force-new-deployment \
  --region $AWS_REGION \
  --output text > /dev/null

aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-streamlit \
  --task-definition cabruca-stg-streamlit \
  --desired-count 1 \
  --force-new-deployment \
  --region $AWS_REGION \
  --output text > /dev/null

echo "Services updated successfully!"

# Check status
echo -e "\n=== Service Status ==="
aws ecs describe-services \
  --cluster cabruca-stg-cluster \
  --services cabruca-stg-api cabruca-stg-streamlit \
  --region $AWS_REGION \
  --query 'services[*].[serviceName,runningCount,desiredCount,pendingCount]' \
  --output table

echo -e "\n=== Deployment Info ==="
echo "API Endpoint: http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/api"
echo "Health Check: http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/health"
echo "Dashboard: http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/dashboard"

echo -e "\nDeployment initiated. Services should be available within 2-3 minutes."
echo "Check logs with: aws logs tail /ecs/cabruca-stg/api --follow --region sa-east-1"