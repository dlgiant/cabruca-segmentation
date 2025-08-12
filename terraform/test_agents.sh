#!/bin/bash

# Test script for Multi-Agent System
# This script runs comprehensive tests on all deployed agents

set -e

echo "========================================="
echo "🧪 Multi-Agent System Test Suite"
echo "========================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Load environment variables
if [ -f ".env.agents" ]; then
    source .env.agents
    echo -e "${GREEN}✅ Environment variables loaded${NC}"
else
    echo -e "${RED}❌ .env.agents not found. Run deploy_agents.sh first.${NC}"
    exit 1
fi

# Test results
TESTS_PASSED=0
TESTS_FAILED=0

# Function to test an agent
test_agent() {
    local agent_name=$1
    local test_payload=$2
    local expected_status=${3:-200}
    
    echo -e "${BLUE}Testing ${agent_name}...${NC}"
    
    # Get the URL variable name
    url_var="${agent_name^^}_URL"
    url="${!url_var}"
    
    if [ -z "$url" ]; then
        echo -e "${RED}  ❌ No URL found for ${agent_name}${NC}"
        ((TESTS_FAILED++))
        return
    fi
    
    # Make the request
    response=$(curl -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        -w "\n%{http_code}" \
        -silent \
        -output /tmp/agent_response.json)
    
    http_code=$(echo "$response" | tail -n 1)
    
    if [ "$http_code" == "$expected_status" ]; then
        echo -e "${GREEN}  ✅ ${agent_name} responded with status ${http_code}${NC}"
        
        # Show response preview
        if [ -f /tmp/agent_response.json ]; then
            echo "  Response preview:"
            jq -r '.' /tmp/agent_response.json 2>/dev/null | head -5 | sed 's/^/    /'
        fi
        
        ((TESTS_PASSED++))
    else
        echo -e "${RED}  ❌ ${agent_name} failed with status ${http_code}${NC}"
        
        # Show error details
        if [ -f /tmp/agent_response.json ]; then
            echo "  Error details:"
            jq -r '.' /tmp/agent_response.json 2>/dev/null | sed 's/^/    /'
        fi
        
        ((TESTS_FAILED++))
    fi
    
    echo ""
}

# Test 1: Manager Agent - Health Check
echo -e "${YELLOW}Test 1: Manager Agent Health Check${NC}"
echo "======================================"
test_agent "manager" '{"action": "health_check"}'

# Test 2: Manager Agent - Simulate Metric Anomaly
echo -e "${YELLOW}Test 2: Manager Agent - Metric Anomaly${NC}"
echo "========================================="
test_agent "manager" '{
    "action": "analyze_metrics",
    "metrics": {
        "cpu_usage": 95,
        "memory_usage": 88,
        "error_rate": 15,
        "latency_p99": 2500
    },
    "threshold": {
        "cpu": 80,
        "memory": 85,
        "error_rate": 5,
        "latency": 1000
    }
}'

# Test 3: Engineer Agent - Feature Request
echo -e "${YELLOW}Test 3: Engineer Agent - Feature Request${NC}"
echo "==========================================="
test_agent "engineer" '{
    "action": "implement_feature",
    "feature": {
        "name": "API Rate Limiting",
        "description": "Add rate limiting to prevent API abuse",
        "priority": "high",
        "requirements": [
            "Limit to 100 requests per minute per IP",
            "Return 429 status when limit exceeded",
            "Add Redis cache for tracking"
        ]
    }
}'

# Test 4: QA Agent - Deployment Validation
echo -e "${YELLOW}Test 4: QA Agent - Deployment Validation${NC}"
echo "==========================================="
test_agent "qa" '{
    "action": "validate_deployment",
    "deployment": {
        "version": "1.2.0",
        "environment": "staging",
        "components": [
            "api-service",
            "worker-service",
            "database-migration"
        ]
    },
    "tests": [
        "health_check",
        "api_contract",
        "database_connectivity",
        "performance_baseline"
    ]
}'

# Test 5: Researcher Agent - Cabruca Analysis
echo -e "${YELLOW}Test 5: Researcher Agent - Cabruca Analysis${NC}"
echo "=============================================="
test_agent "researcher" '{
    "type": "cabruca_analysis",
    "request_id": "test-001",
    "parameters": {
        "region": "bahia",
        "timeframe": "2024-Q1",
        "metrics": ["canopy_coverage", "biodiversity", "productivity"]
    }
}'

# Test 6: Researcher Agent - Anomaly Detection
echo -e "${YELLOW}Test 6: Researcher Agent - Anomaly Detection${NC}"
echo "==============================================="
test_agent "researcher" '{
    "type": "anomaly_detection",
    "request_id": "test-002",
    "simulate_anomaly": true,
    "data_source": "satellite_imagery",
    "threshold": 0.8
}'

# Test 7: Data Processor Agent - Direct Request
echo -e "${YELLOW}Test 7: Data Processor - Cabruca Segmentation${NC}"
echo "================================================"
test_agent "data_processor" '{
    "data_type": "cabruca_segmentation",
    "request_id": "test-003",
    "image_id": "SENTINEL2_20240315_BAHIA",
    "parameters": {
        "model": "unet_cabruca_v2",
        "confidence_threshold": 0.85
    }
}'

# Test 8: Data Processor Agent - Time Series
echo -e "${YELLOW}Test 8: Data Processor - Time Series Analysis${NC}"
echo "================================================"
test_agent "data_processor" '{
    "data_type": "time_series",
    "request_id": "test-004",
    "time_range": "2023-01-01:2024-01-01",
    "metrics": ["vegetation_index", "deforestation_rate"],
    "granularity": "monthly"
}'

# Integration Test: Agent Orchestration
echo -e "${YELLOW}Integration Test: Multi-Agent Workflow${NC}"
echo "========================================="

# Trigger Manager to coordinate other agents
test_agent "manager" '{
    "action": "orchestrate_workflow",
    "workflow": {
        "name": "complete_analysis",
        "steps": [
            {
                "agent": "data_processor",
                "action": "process_satellite_batch",
                "params": {"batch_size": 5}
            },
            {
                "agent": "researcher",
                "action": "analyze_results",
                "depends_on": ["data_processor"]
            },
            {
                "agent": "qa",
                "action": "validate_results",
                "depends_on": ["researcher"]
            }
        ]
    }
}'

# Performance Test: Concurrent Requests
echo -e "${YELLOW}Performance Test: Concurrent Requests${NC}"
echo "========================================"

echo "Sending 10 concurrent requests to Manager Agent..."

for i in {1..10}; do
    (
        curl -X POST "$MANAGER_URL" \
            -H "Content-Type: application/json" \
            -d "{\"action\": \"health_check\", \"request_id\": \"perf-$i\"}" \
            -silent -o /dev/null -w "Request $i: %{http_code} in %{time_total}s\n"
    ) &
done

wait
echo ""

# Check DynamoDB Tables
echo -e "${YELLOW}Verifying DynamoDB Tables${NC}"
echo "=========================="

for table in STATE MEMORY TASKS; do
    table_var="DYNAMODB_${table}_TABLE"
    table_name="${!table_var}"
    
    if [ ! -z "$table_name" ]; then
        item_count=$(aws dynamodb scan --table-name "$table_name" --select COUNT --query 'Count' --output text 2>/dev/null || echo "0")
        echo -e "${GREEN}  ✅ ${table} table: ${item_count} items${NC}"
    else
        echo -e "${RED}  ❌ ${table} table not found${NC}"
    fi
done

echo ""

# Check S3 Buckets
echo -e "${YELLOW}Verifying S3 Buckets${NC}"
echo "====================="

for bucket in ARTIFACTS PROMPTS QUEUE; do
    bucket_var="S3_${bucket}_BUCKET"
    bucket_name="${!bucket_var}"
    
    if [ ! -z "$bucket_name" ]; then
        object_count=$(aws s3 ls "s3://${bucket_name}" --recursive --summarize | grep "Total Objects" | awk '{print $3}' 2>/dev/null || echo "0")
        echo -e "${GREEN}  ✅ ${bucket} bucket: ${object_count} objects${NC}"
    else
        echo -e "${RED}  ❌ ${bucket} bucket not found${NC}"
    fi
done

echo ""

# Check CloudWatch Logs
echo -e "${YELLOW}Checking CloudWatch Logs${NC}"
echo "========================="

for agent in manager engineer qa researcher data_processor; do
    log_group="/aws/lambda/cabruca-segmentation-prod-${agent}-agent"
    
    # Check if log group exists
    if aws logs describe-log-groups --log-group-name-prefix "$log_group" --query "logGroups[0].logGroupName" --output text 2>/dev/null | grep -q "$log_group"; then
        # Get recent log events
        events=$(aws logs filter-log-events \
            --log-group-name "$log_group" \
            --start-time $(($(date +%s) - 300))000 \
            --query 'events | length(@)' \
            --output text 2>/dev/null || echo "0")
        
        echo -e "${GREEN}  ✅ ${agent}: ${events} recent log events${NC}"
    else
        echo -e "${YELLOW}  ⚠️  ${agent}: Log group not found${NC}"
    fi
done

echo ""

# Summary
echo "========================================="
echo -e "${BLUE}📊 Test Summary${NC}"
echo "========================================="
echo -e "${GREEN}Tests Passed: ${TESTS_PASSED}${NC}"
echo -e "${RED}Tests Failed: ${TESTS_FAILED}${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}🎉 All tests passed successfully!${NC}"
    exit 0
else
    echo -e "${RED}⚠️  Some tests failed. Please review the logs.${NC}"
    exit 1
fi