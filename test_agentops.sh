#!/bin/bash

# Test AgentOps integration with Lambda functions
echo "========================================="
echo "🔍 Testing AgentOps Integration"
echo "========================================="
echo ""

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Test each agent multiple times to generate activity
echo -e "${YELLOW}Generating activity for AgentOps tracking...${NC}"
echo ""

# Manager Agent - Multiple invocations
echo -e "${YELLOW}1. Manager Agent - Health Checks${NC}"
for i in {1..3}; do
  echo "  Test $i..."
  aws lambda invoke \
    --function-name cabruca-mvp-mvp-manager-agent \
    --region sa-east-1 \
    --payload '{"action": "health_check", "test_run": '$i'}' \
    /tmp/manager_test_$i.json > /dev/null 2>&1
  sleep 1
done
echo -e "${GREEN}✅ Manager tests completed${NC}"
echo ""

# Engineer Agent - Different actions
echo -e "${YELLOW}2. Engineer Agent - Various Actions${NC}"
aws lambda invoke \
  --function-name cabruca-mvp-mvp-engineer-agent \
  --region sa-east-1 \
  --payload '{"action": "test", "message": "Code review request"}' \
  /tmp/engineer_test_1.json > /dev/null 2>&1

aws lambda invoke \
  --function-name cabruca-mvp-mvp-engineer-agent \
  --region sa-east-1 \
  --payload '{"action": "analyze", "code": "function test() { return true; }"}' \
  /tmp/engineer_test_2.json > /dev/null 2>&1

aws lambda invoke \
  --function-name cabruca-mvp-mvp-engineer-agent \
  --region sa-east-1 \
  --payload '{"action": "optimize", "target": "performance"}' \
  /tmp/engineer_test_3.json > /dev/null 2>&1

echo -e "${GREEN}✅ Engineer tests completed${NC}"
echo ""

# QA Agent - Validation scenarios
echo -e "${YELLOW}3. QA Agent - Validation Tests${NC}"
aws lambda invoke \
  --function-name cabruca-mvp-mvp-qa-agent \
  --region sa-east-1 \
  --payload '{"action": "validate", "deployment": "production", "version": "1.0.0"}' \
  /tmp/qa_test_1.json > /dev/null 2>&1

aws lambda invoke \
  --function-name cabruca-mvp-mvp-qa-agent \
  --region sa-east-1 \
  --payload '{"action": "test_suite", "suite": "integration"}' \
  /tmp/qa_test_2.json > /dev/null 2>&1

echo -e "${GREEN}✅ QA tests completed${NC}"
echo ""

# Researcher Agent - Analysis requests
echo -e "${YELLOW}4. Researcher Agent - Analysis Tasks${NC}"
aws lambda invoke \
  --function-name cabruca-mvp-mvp-researcher-agent \
  --region sa-east-1 \
  --payload '{"type": "cabruca_analysis", "region": "bahia", "year": "2024"}' \
  /tmp/researcher_test_1.json > /dev/null 2>&1

aws lambda invoke \
  --function-name cabruca-mvp-mvp-researcher-agent \
  --region sa-east-1 \
  --payload '{"type": "biodiversity_study", "location": "south_bahia"}' \
  /tmp/researcher_test_2.json > /dev/null 2>&1

echo -e "${GREEN}✅ Researcher tests completed${NC}"
echo ""

# Data Processor - Data processing
echo -e "${YELLOW}5. Data Processor Agent - Processing Tasks${NC}"
aws lambda invoke \
  --function-name cabruca-mvp-mvp-data-processor-agent \
  --region sa-east-1 \
  --payload '{"data_type": "satellite_imagery", "time_range": "2024-Q1", "resolution": "high"}' \
  /tmp/processor_test_1.json > /dev/null 2>&1

aws lambda invoke \
  --function-name cabruca-mvp-mvp-data-processor-agent \
  --region sa-east-1 \
  --payload '{"data_type": "time_series", "time_range": "2024", "metrics": ["ndvi", "evi"]}' \
  /tmp/processor_test_2.json > /dev/null 2>&1

echo -e "${GREEN}✅ Data Processor tests completed${NC}"
echo ""

# Check results
echo "========================================="
echo -e "${YELLOW}Checking AgentOps Tracking Results${NC}"
echo "========================================="
echo ""

# Display sample responses
echo -e "${YELLOW}Sample Response - Manager Agent:${NC}"
cat /tmp/manager_test_1.json | jq '.body' | jq -r '.' | jq '.'

echo ""
echo -e "${YELLOW}Sample Response - Engineer Agent:${NC}"
cat /tmp/engineer_test_1.json | jq '.body' | jq -r '.' | jq '.'

echo ""
echo "========================================="
echo -e "${GREEN}✨ AgentOps Integration Test Complete!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Visit https://app.agentops.ai to view agent activity"
echo "2. Check the Sessions tab for recent agent invocations"
echo "3. Review the Events timeline for detailed tracking"
echo "4. Monitor agent performance metrics"
echo ""
echo "API Key used: ${AGENTOPS_API_KEY:0:10}..."
echo "Environment: MVP"
echo "Region: sa-east-1"