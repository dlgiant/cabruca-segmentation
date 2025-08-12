#!/bin/bash

# Package Enhanced Lambda Functions with GitHub Integration
set -e

echo "==========================================="
echo "📦 Packaging Enhanced Lambda Functions"
echo "==========================================="

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to package a Lambda
package_lambda() {
    local agent_name=$1
    local agent_dir="${agent_name}_agent"
    
    echo -e "${YELLOW}Packaging ${agent_name} agent...${NC}"
    
    # Create package directory
    rm -rf ${agent_dir}/package
    mkdir -p ${agent_dir}/package
    
    # Copy the Lambda function
    cp ${agent_dir}/lambda_function.py ${agent_dir}/package/
    
    # Install minimal dependencies (requests for GitHub API)
    pip install -q requests -t ${agent_dir}/package/ 2>/dev/null || true
    
    # Create ZIP file
    cd ${agent_dir}/package
    zip -q -r ../lambda_function.zip .
    cd ../..
    
    echo -e "${GREEN}✅ ${agent_name} agent packaged${NC}"
}

# Package each agent
package_lambda "manager"
package_lambda "engineer"
package_lambda "qa"

echo ""
echo "==========================================="
echo -e "${GREEN}✨ Enhanced Lambda functions packaged!${NC}"
echo "==========================================="
echo ""
echo "Next steps:"
echo "1. Deploy with: terraform apply -var-file=mvp.tfvars -target=aws_lambda_function.agents"
echo "2. Test with: ./test_pipeline_orchestration.sh"