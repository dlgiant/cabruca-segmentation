#!/bin/bash

# Manager Agent Lambda Deployment Script
# This script packages the Lambda function and its dependencies for deployment

set -e

echo "🚀 Starting Manager Agent Lambda deployment packaging..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Python 3.11 is installed
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}Python 3.11 is required but not installed.${NC}"
    echo "Please install Python 3.11 to continue."
    exit 1
fi

# Clean up previous builds
echo "🧹 Cleaning up previous builds..."
rm -rf package/
rm -f lambda_deployment.zip
rm -f lambda_layer.zip

# Create package directory
mkdir -p package

# Install dependencies for Lambda Layer
echo "📦 Installing dependencies for Lambda Layer..."
pip3.11 install --target ./package/python -r requirements.txt --no-cache-dir

# Create Lambda Layer zip
echo "🗜️ Creating Lambda Layer zip..."
cd package
zip -r ../lambda_layer.zip python -q
cd ..

# Create Lambda deployment package
echo "📦 Creating Lambda deployment package..."
zip lambda_deployment.zip lambda_function.py -q

# Verify the packages
echo "✅ Verifying packages..."
if [ -f "lambda_deployment.zip" ] && [ -f "lambda_layer.zip" ]; then
    echo -e "${GREEN}✓ Lambda deployment package created successfully${NC}"
    echo -e "${GREEN}✓ Lambda layer package created successfully${NC}"
    
    # Show package sizes
    echo ""
    echo "📊 Package sizes:"
    ls -lh lambda_deployment.zip lambda_layer.zip
else
    echo -e "${RED}✗ Failed to create deployment packages${NC}"
    exit 1
fi

# Terraform deployment instructions
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Navigate to the terraform directory: cd terraform/"
echo "2. Initialize Terraform: terraform init"
echo "3. Create a terraform.tfvars file with your Anthropic API key:"
echo "   anthropic_api_key = \"your-api-key-here\""
echo "4. Review the plan: terraform plan"
echo "5. Deploy: terraform apply"
echo ""
echo "🎯 The Lambda function will be scheduled to run every 30 minutes"
echo "💰 Estimated monthly cost: ~$5"

# Clean up temporary package directory
rm -rf package/

echo -e "${GREEN}✨ Deployment packaging complete!${NC}"
