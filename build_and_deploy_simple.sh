#!/bin/bash

set -e

# Force building for AMD64 architecture (for Fargate)
export DOCKER_DEFAULT_PLATFORM=linux/amd64

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
ECR_REPO="cabruca-stg"

echo "Building and deploying simple working container (AMD64 for Fargate)..."

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build simple Docker image
echo "Building Docker image..."
docker build -f Dockerfile.simple -t cabruca-simple .

# Tag and push
echo "Tagging image..."
docker tag cabruca-simple:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:simple-latest

echo "Pushing to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:simple-latest

# Update task definitions to use the new image
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
      "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:simple-latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
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
      "image": "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:simple-latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {"name": "PORT", "value": "8501"}
      ],
      "command": ["python", "dashboard_server.py"],
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

# Register task definitions
echo "Registering task definitions..."
aws ecs register-task-definition --cli-input-json file:///tmp/api-task-def.json --region $AWS_REGION > /dev/null
aws ecs register-task-definition --cli-input-json file:///tmp/streamlit-task-def.json --region $AWS_REGION > /dev/null

# Force new deployment
echo "Updating services..."
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-api --task-definition cabruca-stg-api --force-new-deployment --region $AWS_REGION > /dev/null
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-streamlit --task-definition cabruca-stg-streamlit --force-new-deployment --region $AWS_REGION > /dev/null

echo "Deployment complete! Checking status..."
sleep 10

aws ecs describe-services --cluster cabruca-stg-cluster --services cabruca-stg-api cabruca-stg-streamlit --region $AWS_REGION --query 'services[*].[serviceName,runningCount,desiredCount]' --output table

# Update ALB listener rule for /dashboard
echo -e "\nUpdating ALB listener rules for /dashboard..."
LISTENER_ARN=$(aws elbv2 describe-listeners --load-balancer-arn arn:aws:elasticloadbalancing:sa-east-1:919014037196:loadbalancer/app/cabruca-stg-alb/765edb80e0463b7a --region $AWS_REGION --query 'Listeners[0].ListenerArn' --output text)
STREAMLIT_TG_ARN="arn:aws:elasticloadbalancing:sa-east-1:919014037196:targetgroup/cabruca-stg-streamlit-tg/8f7c88676da0957e"

# Check if dashboard rule exists
DASHBOARD_RULE=$(aws elbv2 describe-rules --listener-arn $LISTENER_ARN --region $AWS_REGION --query "Rules[?Conditions[?PathPatternConfig.Values[?contains(@, '/dashboard')]]].[RuleArn]" --output text 2>/dev/null)

if [ -z "$DASHBOARD_RULE" ]; then
    echo "Creating /dashboard routing rule..."
    aws elbv2 create-rule \
        --listener-arn $LISTENER_ARN \
        --priority 50 \
        --conditions Field=path-pattern,Values="/dashboard*" \
        --actions Type=forward,TargetGroupArn=$STREAMLIT_TG_ARN \
        --region $AWS_REGION \
        --output text > /dev/null
    echo "Dashboard rule created"
else
    echo "Dashboard rule already exists"
fi

echo -e "\nServices will be available at:"
echo "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/health"
echo "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/api"
echo "http://cabruca-stg-alb-428619257.sa-east-1.elb.amazonaws.com/dashboard"