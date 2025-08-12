#!/bin/bash

# Engineer Agent Lambda Deployment Script
# This script packages the Lambda function and its dependencies

set -e

echo "🚀 Starting Engineer Agent Lambda deployment packaging..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
FUNCTION_NAME="engineer-agent"
PYTHON_VERSION="python3.11"
BUILD_DIR="build"
LAYER_DIR="layer"
DEPLOYMENT_PACKAGE="lambda_deployment.zip"
LAYER_PACKAGE="lambda_layer.zip"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -rf $BUILD_DIR $LAYER_DIR
rm -f $DEPLOYMENT_PACKAGE $LAYER_PACKAGE

# Create build directories
mkdir -p $BUILD_DIR
mkdir -p $LAYER_DIR/python

# Package Lambda function
echo "📦 Packaging Lambda function..."
cp lambda_function.py $BUILD_DIR/
cd $BUILD_DIR
zip -r ../$DEPLOYMENT_PACKAGE lambda_function.py
cd ..
echo -e "${GREEN}✓ Lambda function packaged${NC}"

# Install dependencies for Lambda Layer
echo "📚 Installing dependencies for Lambda Layer..."
pip install -r requirements.txt -t $LAYER_DIR/python/ --platform manylinux2014_x86_64 --only-binary=:all: --python-version 3.11

# Remove unnecessary files from layer to reduce size
echo "🔍 Optimizing Lambda Layer..."
find $LAYER_DIR -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find $LAYER_DIR -type d -name "*.dist-info" -exec rm -rf {} + 2>/dev/null || true
find $LAYER_DIR -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true
find $LAYER_DIR -type d -name "test" -exec rm -rf {} + 2>/dev/null || true
find $LAYER_DIR -type f -name "*.pyc" -delete 2>/dev/null || true
find $LAYER_DIR -type f -name "*.pyo" -delete 2>/dev/null || true

# Create Lambda Layer package
echo "📦 Creating Lambda Layer package..."
cd $LAYER_DIR
zip -r ../$LAYER_PACKAGE python/ -q
cd ..
echo -e "${GREEN}✓ Lambda Layer packaged${NC}"

# Check package sizes
DEPLOYMENT_SIZE=$(du -h $DEPLOYMENT_PACKAGE | cut -f1)
LAYER_SIZE=$(du -h $LAYER_PACKAGE | cut -f1)

echo ""
echo "📊 Package Information:"
echo "  • Lambda Function: $DEPLOYMENT_SIZE"
echo "  • Lambda Layer: $LAYER_SIZE"

# Validate package sizes
MAX_LAYER_SIZE=262144000  # 250 MB in bytes
ACTUAL_LAYER_SIZE=$(stat -f%z $LAYER_PACKAGE 2>/dev/null || stat -c%s $LAYER_PACKAGE 2>/dev/null || echo 0)

if [ $ACTUAL_LAYER_SIZE -gt $MAX_LAYER_SIZE ]; then
    echo -e "${RED}⚠️  Warning: Lambda Layer exceeds 250 MB limit!${NC}"
    echo "Consider reducing dependencies or using container images."
else
    echo -e "${GREEN}✓ Package sizes within Lambda limits${NC}"
fi

# Clean up build directories
echo "🧹 Cleaning up build directories..."
rm -rf $BUILD_DIR $LAYER_DIR

echo ""
echo -e "${GREEN}✅ Deployment packages created successfully!${NC}"
echo ""
echo "Next steps:"
echo "1. Navigate to the terraform directory: cd terraform/"
echo "2. Initialize Terraform: terraform init"
echo "3. Plan the deployment: terraform plan -var='github_repo=your-org/your-repo'"
echo "4. Apply the configuration: terraform apply -var='github_repo=your-org/your-repo'"
echo ""
echo "Don't forget to set up the required secrets in AWS Secrets Manager:"
echo "  • github-token (or your custom secret name)"
echo "  • anthropic-api-key (or your custom secret name)"
