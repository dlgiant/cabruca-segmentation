#!/bin/bash

# Test script for deployed multi-agent system

echo "========================================="
echo "🧪 Testing Multi-Agent System Deployment"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get Lambda function names
MANAGER_FUNCTION="cabruca-mvp-mvp-manager-agent"
ENGINEER_FUNCTION="cabruca-mvp-mvp-engineer-agent"
QA_FUNCTION="cabruca-mvp-mvp-qa-agent"
RESEARCHER_FUNCTION="cabruca-mvp-mvp-researcher-agent"
DATA_PROCESSOR_FUNCTION="cabruca-mvp-mvp-data-processor-agent"

echo -e "${YELLOW}Testing Lambda Functions${NC}"
echo "========================="
echo ""

# Test Manager Agent
echo -e "${YELLOW}1. Testing Manager Agent...${NC}"
aws lambda invoke \
  --function-name $MANAGER_FUNCTION \
  --region sa-east-1 \
  --payload '{"action": "health_check"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/manager_response.json 2>&1 | grep -q "StatusCode.*200"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Manager Agent: OK${NC}"
  cat /tmp/manager_response.json | jq '.'
else
  echo -e "${RED}❌ Manager Agent: Failed${NC}"
fi
echo ""

# Test Engineer Agent
echo -e "${YELLOW}2. Testing Engineer Agent...${NC}"
aws lambda invoke \
  --function-name $ENGINEER_FUNCTION \
  --region sa-east-1 \
  --payload '{"action": "test", "message": "Hello Engineer"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/engineer_response.json 2>&1 | grep -q "StatusCode.*200"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Engineer Agent: OK${NC}"
  cat /tmp/engineer_response.json | jq '.'
else
  echo -e "${RED}❌ Engineer Agent: Failed${NC}"
fi
echo ""

# Test QA Agent
echo -e "${YELLOW}3. Testing QA Agent...${NC}"
aws lambda invoke \
  --function-name $QA_FUNCTION \
  --region sa-east-1 \
  --payload '{"action": "validate", "deployment": "test"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/qa_response.json 2>&1 | grep -q "StatusCode.*200"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ QA Agent: OK${NC}"
  cat /tmp/qa_response.json | jq '.'
else
  echo -e "${RED}❌ QA Agent: Failed${NC}"
fi
echo ""

# Test Researcher Agent
echo -e "${YELLOW}4. Testing Researcher Agent...${NC}"
aws lambda invoke \
  --function-name $RESEARCHER_FUNCTION \
  --region sa-east-1 \
  --payload '{"type": "cabruca_analysis", "region": "bahia"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/researcher_response.json 2>&1 | grep -q "StatusCode.*200"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Researcher Agent: OK${NC}"
  cat /tmp/researcher_response.json | jq '.'
else
  echo -e "${RED}❌ Researcher Agent: Failed${NC}"
fi
echo ""

# Test Data Processor Agent
echo -e "${YELLOW}5. Testing Data Processor Agent...${NC}"
aws lambda invoke \
  --function-name $DATA_PROCESSOR_FUNCTION \
  --region sa-east-1 \
  --payload '{"data_type": "time_series", "time_range": "2024"}' \
  --cli-binary-format raw-in-base64-out \
  /tmp/data_processor_response.json 2>&1 | grep -q "StatusCode.*200"

if [ $? -eq 0 ]; then
  echo -e "${GREEN}✅ Data Processor Agent: OK${NC}"
  cat /tmp/data_processor_response.json | jq '.'
else
  echo -e "${RED}❌ Data Processor Agent: Failed${NC}"
fi
echo ""

# Check DynamoDB Tables
echo -e "${YELLOW}Checking DynamoDB Tables${NC}"
echo "========================"

for table in cabruca-mvp-mvp-agent-state cabruca-mvp-mvp-agent-memory cabruca-mvp-mvp-agent-tasks; do
  count=$(aws dynamodb scan --table-name $table --region sa-east-1 --select COUNT --query Count --output text 2>/dev/null)
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ $table: $count items${NC}"
  else
    echo -e "${RED}❌ $table: Not accessible${NC}"
  fi
done
echo ""

# Check S3 Buckets
echo -e "${YELLOW}Checking S3 Buckets${NC}"
echo "==================="

for bucket in cabruca-mvp-mvp-agent-artifacts-919014037196 cabruca-mvp-mvp-agent-prompts-919014037196 cabruca-mvp-mvp-agent-queue-919014037196; do
  count=$(aws s3 ls s3://$bucket --region sa-east-1 --recursive 2>/dev/null | wc -l)
  if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ $bucket: $count objects${NC}"
  else
    echo -e "${RED}❌ $bucket: Not accessible${NC}"
  fi
done
echo ""

# Show Dashboard URLs
echo -e "${YELLOW}Dashboard URLs${NC}"
echo "=============="
echo "CloudWatch Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=sa-east-1#dashboards:name=cabruca-mvp-mvp-agents-dashboard"
echo "AgentOps Dashboard: https://app.agentops.ai"
echo "Cost Control Dashboard: https://console.aws.amazon.com/cloudwatch/home?region=sa-east-1#dashboards:name=mvp-cost-control-dashboard"
echo ""

echo "========================================="
echo -e "${GREEN}✨ Deployment Test Complete!${NC}"
echo "========================================="