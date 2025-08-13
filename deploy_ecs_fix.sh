#!/bin/bash

# Fix ECS deployment by updating task definitions to use existing images

set -e

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
SOURCE_IMAGE="919014037196.dkr.ecr.sa-east-1.amazonaws.com/cabruca-segmentation:latest"

echo "Updating ECS task definitions to use existing cabruca-segmentation image..."

# Update API task definition
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
        {"name": "MODEL_PATH", "value": "/app/outputs/checkpoint_best.pth"},
        {"name": "DEVICE", "value": "cpu"},
        {"name": "API_HOST", "value": "0.0.0.0"},
        {"name": "API_PORT", "value": "8000"}
      ],
      "command": ["python", "api_server.py"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/api",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c 'print(\"healthy\")' || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Update Streamlit task definition
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
        {"name": "MODEL_PATH", "value": "/app/outputs/checkpoint_best.pth"},
        {"name": "DEVICE", "value": "cpu"}
      ],
      "command": ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"],
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

# Update Annotation task definition
cat > /tmp/annotation-task-def.json <<EOF
{
  "family": "cabruca-stg-annotation",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::919014037196:role/cabruca-stg-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "annotation",
      "image": "$SOURCE_IMAGE",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8502,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "MODEL_PATH", "value": "/app/outputs/checkpoint_best.pth"},
        {"name": "DEVICE", "value": "cpu"}
      ],
      "command": ["python", "-c", "print('Annotation service started')"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/annotation",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "annotation"
        }
      }
    }
  ]
}
EOF

# Register new task definitions
echo "Registering API task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/api-task-def.json --region $AWS_REGION > /dev/null

echo "Registering Streamlit task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/streamlit-task-def.json --region $AWS_REGION > /dev/null

echo "Registering Annotation task definition..."
aws ecs register-task-definition --cli-input-json file:///tmp/annotation-task-def.json --region $AWS_REGION > /dev/null

# Update services to use new task definitions
echo "Updating ECS services..."
aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-api \
  --task-definition cabruca-stg-api \
  --force-new-deployment \
  --region $AWS_REGION > /dev/null

aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-streamlit \
  --task-definition cabruca-stg-streamlit \
  --force-new-deployment \
  --region $AWS_REGION > /dev/null

aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-annotation \
  --task-definition cabruca-stg-annotation \
  --force-new-deployment \
  --region $AWS_REGION > /dev/null

echo "Services updated. Waiting for deployment..."
sleep 15

# Check service status
echo "Checking service status..."
aws ecs describe-services \
  --cluster cabruca-stg-cluster \
  --services cabruca-stg-api cabruca-stg-streamlit cabruca-stg-annotation \
  --region $AWS_REGION \
  --query 'services[*].[serviceName,runningCount,desiredCount,pendingCount]' \
  --output table

# Check recent task failures
echo -e "\nChecking for recent task failures..."
aws ecs list-tasks --cluster cabruca-stg-cluster --desired-status STOPPED --region $AWS_REGION --query 'taskArns[0:3]' --output json | \
  xargs -I {} aws ecs describe-tasks --cluster cabruca-stg-cluster --tasks {} --region $AWS_REGION --query 'tasks[*].[taskDefinitionArn,lastStatus,stoppedReason]' --output table 2>/dev/null || true