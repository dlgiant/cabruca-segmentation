#!/bin/bash

# Multi-Agent Pipeline Orchestration Test Script
# This script simulates pipeline issues and tests the multi-agent orchestration

set -e

echo "==========================================="
echo "🔧 Multi-Agent Pipeline Orchestration Test"
echo "==========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get manager agent URL
MANAGER_URL=$(terraform output -json agent_lambda_functions | jq -r '.manager.invoke_url' 2>/dev/null || echo "")

if [ -z "$MANAGER_URL" ]; then
    echo -e "${YELLOW}Using direct Lambda invocation instead of URL${NC}"
    USE_LAMBDA=true
    MANAGER_FUNCTION="cabruca-mvp-mvp-manager-agent"
else
    USE_LAMBDA=false
fi

# Function to invoke manager agent
invoke_manager() {
    local payload=$1
    local description=$2
    
    echo -e "\n${YELLOW}Testing: $description${NC}"
    
    if [ "$USE_LAMBDA" = true ]; then
        # Direct Lambda invocation
        echo "$payload" | base64 > /tmp/payload.txt
        aws lambda invoke \
            --region sa-east-1 \
            --function-name "$MANAGER_FUNCTION" \
            --payload file:///tmp/payload.txt \
            /tmp/response.json > /dev/null 2>&1
        
        cat /tmp/response.json | jq -r '.body' | jq '.'
    else
        # HTTP API invocation
        curl -s -X POST "$MANAGER_URL" \
            -H "Content-Type: application/json" \
            -d "$payload" | jq '.'
    fi
}

# Test 1: Simulate test failure
echo -e "\n${GREEN}Test 1: Simulating test failure in pipeline${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "test_status": "failed",
        "test_details": {
            "failed_tests": ["test_user_authentication", "test_data_validation"],
            "total_tests": 45,
            "passed_tests": 43
        },
        "build_status": "passed",
        "quality_score": 85,
        "test_coverage": 75
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "main"
    }
}' "Test failure detection"

sleep 2

# Test 2: Simulate build failure
echo -e "\n${GREEN}Test 2: Simulating build failure in pipeline${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "build_status": "failed",
        "build_details": {
            "error": "Module not found: boto3",
            "stage": "dependency_installation"
        },
        "test_status": "skipped",
        "quality_score": 0
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "develop"
    }
}' "Build failure detection"

sleep 2

# Test 3: Simulate quality issues
echo -e "\n${GREEN}Test 3: Simulating code quality issues${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "test_status": "passed",
        "build_status": "passed",
        "quality_score": 65,
        "quality_details": {
            "complexity": "high",
            "duplications": 15,
            "code_smells": 23
        },
        "test_coverage": 82
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "feature/new-feature"
    }
}' "Quality issues detection"

sleep 2

# Test 4: Simulate security vulnerability
echo -e "\n${GREEN}Test 4: Simulating security vulnerabilities${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "test_status": "passed",
        "build_status": "passed",
        "quality_score": 90,
        "security_vulnerabilities": 3,
        "security_details": {
            "critical": 1,
            "high": 1,
            "medium": 1,
            "vulnerabilities": [
                "SQL Injection in user input",
                "Hardcoded credentials",
                "Insecure random number generation"
            ]
        },
        "test_coverage": 85
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "main"
    }
}' "Security vulnerability detection"

sleep 2

# Test 5: Simulate low test coverage
echo -e "\n${GREEN}Test 5: Simulating low test coverage${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "test_status": "passed",
        "build_status": "passed",
        "quality_score": 85,
        "test_coverage": 45,
        "coverage_details": {
            "lines_covered": 450,
            "total_lines": 1000,
            "uncovered_files": ["handlers/auth.py", "utils/crypto.py"]
        }
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "main"
    }
}' "Low test coverage detection"

sleep 2

# Test 6: Healthy pipeline (no issues)
echo -e "\n${GREEN}Test 6: Testing healthy pipeline (no issues)${NC}"
invoke_manager '{
    "action": "monitor_pipeline",
    "pipeline_data": {
        "test_status": "passed",
        "build_status": "passed",
        "quality_score": 92,
        "test_coverage": 88,
        "security_vulnerabilities": 0
    },
    "repo_info": {
        "owner": "dlgiant",
        "repo": "cabruca-segmentation",
        "branch": "main"
    }
}' "Healthy pipeline"

# Check workflows created in DynamoDB
echo -e "\n${GREEN}Checking created workflows in DynamoDB...${NC}"
aws dynamodb scan \
    --table-name "cabruca-mvp-mvp-agent-tasks" \
    --region sa-east-1 \
    --max-items 5 \
    --query 'Items[*].{WorkflowID:workflow_id.S, Status:status.S, Issue:issue.M.type.S, Agents:agents_assigned.L[*].S}' \
    --output table 2>/dev/null || echo "No workflows found or table doesn't exist"

echo -e "\n${GREEN}✅ Pipeline orchestration test complete!${NC}"
echo ""
echo "Summary:"
echo "- Manager agent detects various pipeline issues"
echo "- Orchestrates fixes by assigning tasks to engineer and QA agents"
echo "- Creates workflows for tracking progress"
echo ""
echo "Next steps:"
echo "1. Check CloudWatch logs for agent activity"
echo "2. Monitor EventBridge for inter-agent communication"
echo "3. Review mock PRs that would be created (GitHub token required for real PRs)"