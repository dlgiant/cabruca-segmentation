#!/bin/bash

# Unified Deployment Script
# Automatically detects environment based on current git branch

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Cabruca Segmentation Deployment Script"
echo "========================================="

# Detect current branch
CURRENT_BRANCH=$(git branch --show-current)
echo "Current branch: $CURRENT_BRANCH"

# Determine environment based on branch
if [ "$CURRENT_BRANCH" = "main" ]; then
    ENVIRONMENT="production"
    CONFIG_FILE="configs/production.env"
    echo -e "${GREEN}Deploying to PRODUCTION environment${NC}"
    
    # Confirm production deployment
    echo -e "${YELLOW}⚠️  WARNING: You are about to deploy to PRODUCTION!${NC}"
    read -p "Are you sure? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Deployment cancelled"
        exit 0
    fi
else
    ENVIRONMENT="staging"
    CONFIG_FILE="configs/staging.env"
    echo -e "${GREEN}Deploying to STAGING environment${NC}"
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}Error: Configuration file $CONFIG_FILE not found${NC}"
    exit 1
fi

# Load environment configuration
source $CONFIG_FILE
echo "Loaded configuration from: $CONFIG_FILE"

# Export for Docker
export DOCKER_DEFAULT_PLATFORM=linux/amd64

# Login to ECR
echo "Logging into AWS ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo "Building Docker image for $ENVIRONMENT..."
if [ "$ENVIRONMENT" = "production" ]; then
    # Use full Dockerfile for production
    docker build -t cabruca-$ENVIRONMENT .
else
    # Use simple Dockerfile for staging
    docker build -f Dockerfile.simple -t cabruca-$ENVIRONMENT .
fi

# Tag image
IMAGE_TAG=$(git rev-parse --short HEAD)
echo "Tagging image with: $IMAGE_TAG"
docker tag cabruca-$ENVIRONMENT:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker tag cabruca-$ENVIRONMENT:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker tag cabruca-$ENVIRONMENT:latest \
    $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$ENVIRONMENT

# Push to ECR
echo "Pushing image to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$IMAGE_TAG
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:$ENVIRONMENT

# Update ECS task definitions
echo "Updating ECS task definitions..."

# Create task definition for API
cat > /tmp/api-task-def.json <<EOF
{
  "family": "${ECS_SERVICE_API}-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${TASK_CPU}",
  "memory": "${TASK_MEMORY}",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/cabruca-${ENVIRONMENT}-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/cabruca-${ENVIRONMENT}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "api",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "${ENVIRONMENT}"},
        {"name": "ENABLE_DEBUG_MODE", "value": "${ENABLE_DEBUG_MODE}"},
        {"name": "ENABLE_CACHE", "value": "${ENABLE_CACHE}"},
        {"name": "CACHE_TTL", "value": "${CACHE_TTL}"}
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${ECS_CLUSTER}/api",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "api"
        }
      }
    }
  ]
}
EOF

# Create task definition for Streamlit
cat > /tmp/streamlit-task-def.json <<EOF
{
  "family": "${ECS_SERVICE_STREAMLIT}-task",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "${TASK_CPU}",
  "memory": "${TASK_MEMORY}",
  "executionRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/cabruca-${ENVIRONMENT}-ecs-task-execution-role",
  "taskRoleArn": "arn:aws:iam::${AWS_ACCOUNT_ID}:role/cabruca-${ENVIRONMENT}-ecs-task-role",
  "containerDefinitions": [
    {
      "name": "streamlit",
      "image": "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}:${IMAGE_TAG}",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "ENVIRONMENT", "value": "${ENVIRONMENT}"},
        {"name": "API_URL", "value": "http://${ALB_DNS}"}
      ],
      "command": ["python", "dashboard_server.py"],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/${ECS_CLUSTER}/streamlit",
          "awslogs-region": "${AWS_REGION}",
          "awslogs-stream-prefix": "streamlit"
        }
      }
    }
  ]
}
EOF

# Register task definitions
echo "Registering task definitions..."
aws ecs register-task-definition \
    --cli-input-json file:///tmp/api-task-def.json \
    --region $AWS_REGION > /dev/null

aws ecs register-task-definition \
    --cli-input-json file:///tmp/streamlit-task-def.json \
    --region $AWS_REGION > /dev/null

# Update ECS services
echo "Updating ECS services..."
aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE_API \
    --task-definition ${ECS_SERVICE_API}-task \
    --desired-count $DESIRED_COUNT \
    --force-new-deployment \
    --region $AWS_REGION > /dev/null

aws ecs update-service \
    --cluster $ECS_CLUSTER \
    --service $ECS_SERVICE_STREAMLIT \
    --task-definition ${ECS_SERVICE_STREAMLIT}-task \
    --desired-count $DESIRED_COUNT \
    --force-new-deployment \
    --region $AWS_REGION > /dev/null

echo "Waiting for services to stabilize..."
sleep 30

# Check deployment status
echo "Checking deployment status..."
aws ecs describe-services \
    --cluster $ECS_CLUSTER \
    --services $ECS_SERVICE_API $ECS_SERVICE_STREAMLIT \
    --region $AWS_REGION \
    --query 'services[*].[serviceName,runningCount,desiredCount]' \
    --output table

# Health checks
echo "Running health checks..."
HEALTH_CHECK_URL="http://$ALB_DNS/health"
API_CHECK_URL="http://$ALB_DNS/api"
DASHBOARD_CHECK_URL="http://$ALB_DNS/dashboard"

if curl -f -s -o /dev/null -w "%{http_code}" $HEALTH_CHECK_URL | grep -q "200"; then
    echo -e "${GREEN}✅ Health check passed${NC}"
else
    echo -e "${RED}❌ Health check failed${NC}"
fi

if curl -f -s -o /dev/null -w "%{http_code}" $API_CHECK_URL | grep -q "200"; then
    echo -e "${GREEN}✅ API check passed${NC}"
else
    echo -e "${RED}❌ API check failed${NC}"
fi

if curl -f -s -o /dev/null -w "%{http_code}" $DASHBOARD_CHECK_URL | grep -q "200"; then
    echo -e "${GREEN}✅ Dashboard check passed${NC}"
else
    echo -e "${RED}❌ Dashboard check failed${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}Deployment to $ENVIRONMENT complete!${NC}"
echo ""
echo "Service URLs:"
echo "  Health: http://$ALB_DNS/health"
echo "  API: http://$ALB_DNS/api"
echo "  Dashboard: http://$ALB_DNS/dashboard"
echo "  Streamlit: http://$ALB_DNS/streamlit"
echo ""
echo "Deployment details:"
echo "  Environment: $ENVIRONMENT"
echo "  Image tag: $IMAGE_TAG"
echo "  Cluster: $ECS_CLUSTER"
echo "  Desired count: $DESIRED_COUNT"
echo "========================================="