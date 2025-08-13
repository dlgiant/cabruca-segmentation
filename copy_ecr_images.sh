#!/bin/bash

# Copy existing images from cabruca-segmentation to cabruca-stg repository

set -e

AWS_REGION="sa-east-1"
AWS_ACCOUNT_ID="919014037196"
SOURCE_REPO="cabruca-segmentation"
DEST_REPO="cabruca-stg"

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Pull the latest image from source repository
echo "Pulling latest image from $SOURCE_REPO..."
docker pull $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SOURCE_REPO:latest

# Tag for destination repository with multiple service tags
echo "Tagging images for $DEST_REPO..."
docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SOURCE_REPO:latest \
           $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:api-latest

docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SOURCE_REPO:latest \
           $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:streamlit-latest

docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SOURCE_REPO:latest \
           $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:annotation-latest

docker tag $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$SOURCE_REPO:latest \
           $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:latest

# Push all tagged images
echo "Pushing images to $DEST_REPO..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:api-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:streamlit-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:annotation-latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$DEST_REPO:latest

echo "Images copied successfully!"

# Force new deployments on ECS services
echo "Forcing new deployments on ECS services..."
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-api --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-streamlit --force-new-deployment --region $AWS_REGION
aws ecs update-service --cluster cabruca-stg-cluster --service cabruca-stg-annotation --force-new-deployment --region $AWS_REGION

echo "ECS services updated. Checking status..."
sleep 10

# Check service status
aws ecs describe-services --cluster cabruca-stg-cluster --services cabruca-stg-api cabruca-stg-streamlit cabruca-stg-annotation --region $AWS_REGION --query 'services[*].[serviceName,runningCount,desiredCount,pendingCount]' --output table