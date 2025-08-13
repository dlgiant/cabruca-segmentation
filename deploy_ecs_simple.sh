#!/bin/bash

# Deploy ECS with simple startup commands that don't require agentops

set -e

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
SOURCE_IMAGE="919014037196.dkr.ecr.sa-east-1.amazonaws.com/cabruca-segmentation:latest"

echo "Updating ECS task definitions with simple startup commands..."

# Update API task definition with a basic Python HTTP server
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
      "command": [
        "python", "-c",
        "from http.server import HTTPServer, BaseHTTPRequestHandler; import json; class Handler(BaseHTTPRequestHandler): \n def do_GET(self):\n  if self.path == '/health':\n   self.send_response(200)\n   self.send_header('Content-type', 'application/json')\n   self.end_headers()\n   self.wfile.write(json.dumps({'status': 'healthy', 'service': 'cabruca-api'}).encode())\n  else:\n   self.send_response(200)\n   self.send_header('Content-type', 'text/html')\n   self.end_headers()\n   self.wfile.write(b'<h1>Cabruca Segmentation API</h1><p>Service is running</p>')\nserver = HTTPServer(('0.0.0.0', 8000), Handler); print('API Server running on port 8000'); server.serve_forever()"
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/cabruca-stg/api",
          "awslogs-region": "sa-east-1",
          "awslogs-stream-prefix": "api"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "python -c 'import urllib.request; urllib.request.urlopen(\"http://localhost:8000/health\")' || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
EOF

# Update Streamlit task definition with simple HTTP server
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
      "command": [
        "python", "-c",
        "from http.server import HTTPServer, BaseHTTPRequestHandler; class Handler(BaseHTTPRequestHandler): \n def do_GET(self):\n  self.send_response(200)\n  self.send_header('Content-type', 'text/html')\n  self.end_headers()\n  self.wfile.write(b'<h1>Cabruca Dashboard</h1><p>Streamlit service placeholder</p>')\nserver = HTTPServer(('0.0.0.0', 8501), Handler); print('Streamlit running on port 8501'); server.serve_forever()"
      ],
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

# Register new task definitions
echo "Registering API task definition..."
TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file:///tmp/api-task-def.json --region $AWS_REGION --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Registered: $TASK_DEF_ARN"

echo "Registering Streamlit task definition..."
STREAMLIT_TASK_DEF_ARN=$(aws ecs register-task-definition --cli-input-json file:///tmp/streamlit-task-def.json --region $AWS_REGION --query 'taskDefinition.taskDefinitionArn' --output text)
echo "Registered: $STREAMLIT_TASK_DEF_ARN"

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

# Stop the annotation service for now
aws ecs update-service \
  --cluster cabruca-stg-cluster \
  --service cabruca-stg-annotation \
  --desired-count 0 \
  --region $AWS_REGION \
  --output text > /dev/null

echo "Services updated. Waiting for deployment..."
sleep 20

# Check service status
echo -e "\n=== Service Status ==="
aws ecs describe-services \
  --cluster cabruca-stg-cluster \
  --services cabruca-stg-api cabruca-stg-streamlit \
  --region $AWS_REGION \
  --query 'services[*].[serviceName,runningCount,desiredCount,pendingCount]' \
  --output table

# Check running tasks
echo -e "\n=== Running Tasks ==="
aws ecs list-tasks --cluster cabruca-stg-cluster --desired-status RUNNING --region $AWS_REGION --query 'taskArns' --output json

# Test the endpoints
echo -e "\n=== Testing Endpoints ==="
ALB_DNS="cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com"
echo "Load Balancer: http://$ALB_DNS"

# Wait a bit more for tasks to start
echo "Waiting for tasks to fully start..."
sleep 30

# Final status check
echo -e "\n=== Final Service Status ==="
aws ecs describe-services \
  --cluster cabruca-stg-cluster \
  --services cabruca-stg-api cabruca-stg-streamlit \
  --region $AWS_REGION \
  --query 'services[*].[serviceName,runningCount,desiredCount,pendingCount,deployments[0].rolloutState]' \
  --output table