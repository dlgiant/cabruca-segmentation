#!/bin/bash

# Deployment script for Multi-Agent System
# This script packages Lambda functions and deploys the infrastructure

set -e  # Exit on error

echo "========================================="
echo "🚀 Multi-Agent System Deployment Script"
echo "========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Load environment variables from .env if it exists
if [ -f "../.env" ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    source ../.env
elif [ -f ".env" ]; then
    echo -e "${GREEN}Loading environment variables from .env${NC}"
    source .env
fi

# Configuration
ENVIRONMENT="${ENVIRONMENT:-prod}"
AWS_REGION="${AWS_REGION:-sa-east-1}"
TERRAFORM_DIR="$(pwd)"

echo -e "${YELLOW}Environment: ${ENVIRONMENT}${NC}"
echo -e "${YELLOW}AWS Region: ${AWS_REGION}${NC}"
echo ""

# Function to package Lambda
package_lambda() {
    local agent_name=$1
    local agent_dir="${agent_name//-/_}"
    
    echo -e "${YELLOW}📦 Packaging ${agent_name}...${NC}"
    
    if [ -d "${agent_dir}" ]; then
        cd "${agent_dir}"
        
        # Create package directory
        rm -rf package
        mkdir -p package
        
        # Install dependencies
        if [ -f "requirements.txt" ]; then
            pip install -r requirements.txt -t package/ --quiet
        fi
        
        # Copy function code
        cp lambda_function.py package/
        
        # Create ZIP
        cd package
        zip -r9 ../lambda_function.zip . > /dev/null 2>&1
        cd ..
        
        # Clean up
        rm -rf package
        
        echo -e "${GREEN}✅ ${agent_name} packaged successfully${NC}"
        cd "${TERRAFORM_DIR}"
    else
        echo -e "${RED}❌ Directory ${agent_dir} not found${NC}"
    fi
}

# Step 1: Package all Lambda functions
echo -e "${GREEN}Step 1: Packaging Lambda Functions${NC}"
echo "======================================"

package_lambda "manager-agent"
package_lambda "engineer-agent"
package_lambda "qa-agent"
package_lambda "researcher-agent"
package_lambda "data-processor-agent"

echo ""

# Step 2: Initialize Terraform
echo -e "${GREEN}Step 2: Initializing Terraform${NC}"
echo "================================="

terraform init -upgrade

echo ""

# Step 3: Validate Terraform configuration
echo -e "${GREEN}Step 3: Validating Configuration${NC}"
echo "===================================="

terraform validate

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Terraform configuration is valid${NC}"
else
    echo -e "${RED}❌ Terraform validation failed${NC}"
    exit 1
fi

echo ""

# Step 4: Create terraform plan
echo -e "${GREEN}Step 4: Creating Terraform Plan${NC}"
echo "=================================="

if [ -f "mvp.tfvars" ]; then
    terraform plan -var-file="mvp.tfvars" -out=tfplan
else
    echo -e "${YELLOW}⚠️  mvp.tfvars not found, using defaults${NC}"
    terraform plan -out=tfplan
fi

echo ""

# Step 5: Apply infrastructure
echo -e "${GREEN}Step 5: Applying Infrastructure${NC}"
echo "=================================="
echo -e "${YELLOW}This will create real AWS resources and incur costs.${NC}"
read -p "Do you want to proceed? (yes/no): " confirm

if [ "$confirm" == "yes" ]; then
    terraform apply tfplan
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Infrastructure deployed successfully!${NC}"
    else
        echo -e "${RED}❌ Deployment failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Deployment cancelled${NC}"
    exit 0
fi

echo ""

# Step 6: Output important information
echo -e "${GREEN}Step 6: Deployment Information${NC}"
echo "================================"

# Get outputs
terraform output -json > deployment_output.json

echo -e "${GREEN}Key Resources Created:${NC}"
echo "----------------------"

# Parse and display outputs
if command -v jq &> /dev/null; then
    echo "Lambda Functions:"
    jq -r '.agent_lambda_functions.value | to_entries[] | "  - \(.key): \(.value.invoke_url)"' deployment_output.json
    
    echo ""
    echo "DynamoDB Tables:"
    jq -r '.agent_dynamodb_tables.value | to_entries[] | "  - \(.key): \(.value)"' deployment_output.json
    
    echo ""
    echo "S3 Buckets:"
    jq -r '.agent_s3_buckets.value | to_entries[] | "  - \(.key): \(.value)"' deployment_output.json
    
    echo ""
    echo "Monitoring:"
    jq -r '.agent_cloudwatch_dashboard.value' deployment_output.json | xargs echo "  - Dashboard URL:"
else
    cat deployment_output.json
fi

echo ""

# Step 7: Configure environment variables
echo -e "${GREEN}Step 7: Configuring Environment Variables${NC}"
echo "==========================================="

# Create .env file for local testing
cat > .env.agents << EOF
# Agent Environment Variables
export AWS_REGION=${AWS_REGION}
export ENVIRONMENT=${ENVIRONMENT}
export AGENTOPS_API_KEY=${AGENTOPS_API_KEY:-your-api-key-here}

# Lambda Function URLs
$(terraform output -json | jq -r '.agent_lambda_functions.value | to_entries[] | "export \(.key | ascii_upcase)_URL=\(.value.invoke_url)"')

# DynamoDB Tables
$(terraform output -json | jq -r '.agent_dynamodb_tables.value | to_entries[] | "export DYNAMODB_\(.key | ascii_upcase)_TABLE=\(.value)"')

# S3 Buckets
$(terraform output -json | jq -r '.agent_s3_buckets.value | to_entries[] | "export S3_\(.key | ascii_upcase)_BUCKET=\(.value)"')
EOF

echo -e "${GREEN}✅ Environment variables saved to .env.agents${NC}"
echo -e "${YELLOW}Source it with: source .env.agents${NC}"

echo ""

# Step 8: Run initial tests
echo -e "${GREEN}Step 8: Running Initial Tests${NC}"
echo "================================"

# Test Manager Agent
echo -e "${YELLOW}Testing Manager Agent...${NC}"
MANAGER_URL=$(terraform output -json | jq -r '.agent_lambda_functions.value.manager.invoke_url')

if [ ! -z "$MANAGER_URL" ]; then
    curl -X POST "$MANAGER_URL" \
        -H "Content-Type: application/json" \
        -d '{"action": "health_check"}' \
        --silent --show-error | jq '.' || echo "Test failed"
fi

echo ""
echo "========================================="
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo "========================================="
echo ""
echo "Next Steps:"
echo "1. Configure AgentOps API key in Lambda environment variables"
echo "2. Run the test suite: ./test_agents.sh"
echo "3. Monitor the CloudWatch dashboard"
echo "4. Check the incident response runbook"
echo ""
echo -e "${YELLOW}To destroy resources: terraform destroy -var-file=mvp.tfvars${NC}"