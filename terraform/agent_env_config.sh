#!/bin/bash

# Environment Configuration Script for Multi-Agent System
# This script configures environment variables for all Lambda functions

set -e

echo "========================================="
echo "🔧 Configuring Agent Environment Variables"
echo "========================================="

# Check if AWS CLI is configured
if ! aws sts get-caller-identity > /dev/null 2>&1; then
    echo "❌ AWS CLI not configured. Please run 'aws configure'"
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REGION=${AWS_REGION:-sa-east-1}
ENVIRONMENT=${1:-prod}

echo "Account ID: $ACCOUNT_ID"
echo "Region: $REGION"
echo "Environment: $ENVIRONMENT"
echo ""

# Function to update Lambda environment variables
update_lambda_env() {
    local function_name=$1
    local env_vars=$2
    
    echo "Updating $function_name..."
    
    aws lambda update-function-configuration \
        --function-name "$function_name" \
        --environment "Variables=$env_vars" \
        --output text > /dev/null
    
    if [ $? -eq 0 ]; then
        echo "  ✅ $function_name updated"
    else
        echo "  ❌ Failed to update $function_name"
    fi
}

# Common environment variables for all agents
COMMON_ENV='{
    "AWS_REGION":"'$REGION'",
    "ENVIRONMENT":"'$ENVIRONMENT'",
    "LOG_LEVEL":"INFO",
    "AGENTOPS_API_KEY":"'${AGENTOPS_API_KEY:-}'",
    "DYNAMODB_STATE_TABLE":"cabruca-segmentation-'$ENVIRONMENT'-agent-state",
    "DYNAMODB_MEMORY_TABLE":"cabruca-segmentation-'$ENVIRONMENT'-agent-memory",
    "DYNAMODB_TASKS_TABLE":"cabruca-segmentation-'$ENVIRONMENT'-agent-tasks",
    "S3_ARTIFACTS_BUCKET":"cabruca-segmentation-'$ENVIRONMENT'-agent-artifacts-'$ACCOUNT_ID'",
    "S3_PROMPTS_BUCKET":"cabruca-segmentation-'$ENVIRONMENT'-agent-prompts-'$ACCOUNT_ID'",
    "S3_QUEUE_BUCKET":"cabruca-segmentation-'$ENVIRONMENT'-agent-queue-'$ACCOUNT_ID'",
    "S3_DATA_BUCKET":"cabruca-segmentation-'$ENVIRONMENT'-data-brasil",
    "S3_MODELS_BUCKET":"cabruca-segmentation-'$ENVIRONMENT'-models-brasil",
    "EVENTBRIDGE_BUS":"default",
    "MAX_RETRIES":"3",
    "TIMEOUT_SECONDS":"300"
}'

# Agent-specific environment variables
MANAGER_ENV=$(echo $COMMON_ENV | jq '. + {
    "AGENT_TYPE":"MANAGER",
    "ORCHESTRATION_ENABLED":"true",
    "MAX_CONCURRENT_WORKFLOWS":"5"
}' | jq -c .)

ENGINEER_ENV=$(echo $COMMON_ENV | jq '. + {
    "AGENT_TYPE":"ENGINEER",
    "CODE_GENERATION_MODEL":"gpt-4",
    "MAX_CODE_LENGTH":"10000"
}' | jq -c .)

QA_ENV=$(echo $COMMON_ENV | jq '. + {
    "AGENT_TYPE":"QA",
    "TEST_ENDPOINTS":"'${API_ENDPOINT:-}'",
    "COVERAGE_THRESHOLD":"80",
    "PERFORMANCE_BASELINE":"1000"
}' | jq -c .)

RESEARCHER_ENV=$(echo $COMMON_ENV | jq '. + {
    "AGENT_TYPE":"RESEARCHER",
    "ANALYSIS_DEPTH":"comprehensive",
    "DATA_SOURCES":"satellite,weather,soil,historical"
}' | jq -c .)

DATA_PROCESSOR_ENV=$(echo $COMMON_ENV | jq '. + {
    "AGENT_TYPE":"DATA_PROCESSOR",
    "BATCH_SIZE":"100",
    "PARALLEL_PROCESSING":"true",
    "IMAGE_FORMATS":"tif,tiff,png,jpg"
}' | jq -c .)

# Update each Lambda function
echo "Updating Lambda environment variables..."
echo "========================================"

update_lambda_env "cabruca-segmentation-$ENVIRONMENT-manager-agent" "$MANAGER_ENV"
update_lambda_env "cabruca-segmentation-$ENVIRONMENT-engineer-agent" "$ENGINEER_ENV"
update_lambda_env "cabruca-segmentation-$ENVIRONMENT-qa-agent" "$QA_ENV"
update_lambda_env "cabruca-segmentation-$ENVIRONMENT-researcher-agent" "$RESEARCHER_ENV"
update_lambda_env "cabruca-segmentation-$ENVIRONMENT-data-processor-agent" "$DATA_PROCESSOR_ENV"

echo ""
echo "========================================="
echo "✅ Environment configuration complete!"
echo "========================================="

# Verify configuration
echo ""
echo "Verifying configuration..."
echo "=========================="

for agent in manager engineer qa researcher data-processor; do
    echo -n "  $agent: "
    
    # Get AGENTOPS_API_KEY from the function
    api_key=$(aws lambda get-function-configuration \
        --function-name "cabruca-segmentation-$ENVIRONMENT-${agent}-agent" \
        --query 'Environment.Variables.AGENTOPS_API_KEY' \
        --output text 2>/dev/null)
    
    if [ "$api_key" != "None" ] && [ ! -z "$api_key" ]; then
        echo "✅ Configured"
    else
        echo "⚠️  Missing AGENTOPS_API_KEY"
    fi
done

echo ""
echo "Note: If AGENTOPS_API_KEY is missing, set it with:"
echo "export AGENTOPS_API_KEY='your-api-key'"
echo "Then run this script again."