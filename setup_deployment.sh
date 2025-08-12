#!/bin/bash

# ========================================
# Cabruca Multi-Agent System Setup Script
# Complete deployment helper for steps 2-5
# ========================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "========================================="
echo -e "${CYAN}🚀 Cabruca Multi-Agent System Setup${NC}"
echo "========================================="
echo ""

# Step 1: Check for .env file
echo -e "${YELLOW}Step 1: Environment Configuration${NC}"
echo "=================================="

if [ ! -f ".env" ]; then
    echo -e "${YELLOW}No .env file found. Creating from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✅ Created .env file from template${NC}"
    echo ""
    echo -e "${RED}IMPORTANT: Please edit the .env file with your actual values:${NC}"
    echo "  1. AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"
    echo "  2. AWS_ACCOUNT_ID (get with: aws sts get-caller-identity)"
    echo "  3. AGENTOPS_API_KEY (from https://app.agentops.ai)"
    echo "  4. ALERT_EMAIL for notifications"
    echo "  5. API_KEY and SECRET_KEY for security"
    echo ""
    echo -e "${YELLOW}Opening .env file for editing...${NC}"
    
    # Open in default editor
    if command -v code &> /dev/null; then
        code .env
    elif command -v nano &> /dev/null; then
        nano .env
    else
        vi .env
    fi
    
    echo ""
    read -p "Press enter when you've updated the .env file..."
fi

# Load environment variables
source .env

# Validate required variables
echo -e "${YELLOW}Validating environment variables...${NC}"
MISSING_VARS=()

[ -z "$AWS_ACCESS_KEY_ID" ] && MISSING_VARS+=("AWS_ACCESS_KEY_ID")
[ -z "$AWS_SECRET_ACCESS_KEY" ] && MISSING_VARS+=("AWS_SECRET_ACCESS_KEY")
[ -z "$AGENTOPS_API_KEY" ] && MISSING_VARS+=("AGENTOPS_API_KEY")
[ -z "$ALERT_EMAIL" ] && MISSING_VARS+=("ALERT_EMAIL")

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please update your .env file and run this script again."
    exit 1
fi

echo -e "${GREEN}✅ All required variables are set${NC}"
echo ""

# Get AWS Account ID if not set
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${YELLOW}Getting AWS Account ID...${NC}"
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    echo "AWS_ACCOUNT_ID=$AWS_ACCOUNT_ID" >> .env
    echo -e "${GREEN}✅ AWS Account ID: $AWS_ACCOUNT_ID${NC}"
fi

# Step 2: Deploy Infrastructure
echo ""
echo -e "${YELLOW}Step 2: Deploy Infrastructure${NC}"
echo "=============================="
echo "This will deploy all AWS resources for the multi-agent system."
echo -e "${YELLOW}Estimated monthly cost: \$10-35 USD${NC}"
echo ""
read -p "Do you want to proceed with deployment? (yes/no): " deploy_confirm

if [ "$deploy_confirm" == "yes" ]; then
    cd terraform
    
    # Update mvp.tfvars with values from .env
    echo -e "${YELLOW}Updating Terraform variables...${NC}"
    cat > mvp_deployment.tfvars << EOF
# Auto-generated from .env file
project_name = "$PROJECT_NAME"
environment  = "$ENVIRONMENT"
aws_region   = "$AWS_REGION"

# Alert configuration
alert_email = "$ALERT_EMAIL"
alert_phone = "${ALERT_PHONE:-}"
cost_alert_threshold = ${COST_ALERT_THRESHOLD:-100}
cost_threshold = ${COST_HARD_LIMIT:-150}

# Monitoring
monitoring_configuration = {
  enable_cloudwatch   = true
  enable_xray         = ${ENABLE_XRAY:-true}
  log_retention_days  = 14
  alarm_email         = "$ALERT_EMAIL"
  slack_webhook_url   = "${SLACK_WEBHOOK_URL:-}"
}

# Use existing mvp.tfvars for other settings
EOF
    
    # Run deployment
    echo -e "${YELLOW}Running deployment script...${NC}"
    export AGENTOPS_API_KEY=$AGENTOPS_API_KEY
    ./deploy_agents.sh
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Infrastructure deployed successfully!${NC}"
    else
        echo -e "${RED}❌ Deployment failed. Check the logs above.${NC}"
        exit 1
    fi
    
    cd ..
else
    echo -e "${YELLOW}Skipping deployment. Run './terraform/deploy_agents.sh' when ready.${NC}"
fi

# Step 3: Configure Environment Variables
echo ""
echo -e "${YELLOW}Step 3: Configure Agent Environment Variables${NC}"
echo "=============================================="

cd terraform
./agent_env_config.sh $ENVIRONMENT
cd ..

echo -e "${GREEN}✅ Agent environment configured${NC}"

# Step 4: Test Agents
echo ""
echo -e "${YELLOW}Step 4: Test Multi-Agent System${NC}"
echo "================================="
echo "Running comprehensive test suite..."
echo ""

cd terraform
./test_agents.sh

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ All agent tests passed!${NC}"
else
    echo -e "${YELLOW}⚠️  Some tests failed. Review the output above.${NC}"
fi
cd ..

# Step 5: Setup Monitoring
echo ""
echo -e "${YELLOW}Step 5: Setup Monitoring Dashboard${NC}"
echo "===================================="

# Get CloudWatch Dashboard URL
DASHBOARD_URL=$(cd terraform && terraform output -raw agent_cloudwatch_dashboard 2>/dev/null || echo "")

if [ ! -z "$DASHBOARD_URL" ]; then
    echo -e "${GREEN}CloudWatch Dashboard:${NC}"
    echo "  $DASHBOARD_URL"
    echo ""
fi

echo -e "${CYAN}AgentOps Setup Instructions:${NC}"
echo "1. Go to https://app.agentops.ai"
echo "2. Create a new project called 'Cabruca Segmentation'"
echo "3. Your API key is already configured in the agents"
echo "4. Set up these custom events:"
echo "   - research_completed"
echo "   - processing_completed"
echo "   - orchestration_started"
echo "   - task_failed"
echo ""

# 24-Hour Monitoring Setup
echo -e "${YELLOW}24-Hour Monitoring Checklist:${NC}"
echo "================================"
echo ""
echo -e "${CYAN}Hour 1-4: Initial Validation${NC}"
echo "  [ ] All Lambda functions responding"
echo "  [ ] No error alarms triggered"
echo "  [ ] DynamoDB tables receiving data"
echo "  [ ] S3 buckets accessible"
echo "  [ ] CloudWatch logs streaming"
echo "  [ ] AgentOps showing sessions"
echo ""

echo -e "${CYAN}Hour 4-8: Load Testing${NC}"
echo "  [ ] Run gradual load test (provided below)"
echo "  [ ] Monitor Lambda concurrent executions"
echo "  [ ] Check DynamoDB throttling"
echo "  [ ] Review memory/CPU usage"
echo ""

echo -e "${CYAN}Hour 8-12: Performance Tuning${NC}"
echo "  [ ] Adjust Lambda memory if needed"
echo "  [ ] Tune DynamoDB capacity"
echo "  [ ] Review CloudWatch metrics"
echo "  [ ] Optimize S3 lifecycle policies"
echo ""

echo -e "${CYAN}Hour 12-24: Production Readiness${NC}"
echo "  [ ] Monitor cost trends"
echo "  [ ] Review error patterns"
echo "  [ ] Adjust alarm thresholds"
echo "  [ ] Document any issues"
echo ""

# Create monitoring scripts
echo -e "${YELLOW}Creating monitoring helper scripts...${NC}"

# Cost monitoring script
cat > monitor_costs.sh << 'EOF'
#!/bin/bash
# Daily cost monitoring script

echo "AWS Cost Report - $(date)"
echo "========================="

# Get today's costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d yesterday +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE \
  --output table

# Get month-to-date costs
echo ""
echo "Month-to-Date Total:"
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --query 'ResultsByTime[0].Total.UnblendedCost.Amount' \
  --output text | xargs printf "$%.2f\n"

# Forecast
echo ""
echo "Monthly Forecast:"
aws ce get-cost-forecast \
  --time-period Start=$(date +%Y-%m-%d),End=$(date -d '+1 month' +%Y-%m-01) \
  --metric UNBLENDED_COST \
  --granularity MONTHLY \
  --query 'Total.Amount' \
  --output text | xargs printf "$%.2f\n"
EOF

chmod +x monitor_costs.sh

# Load test script
cat > load_test.sh << 'EOF'
#!/bin/bash
# Gradual load test for agents

source .env

echo "Starting gradual load test..."
echo "============================="

# Test each agent with increasing load
for i in {1..10}; do
    echo "Load level: $i/10"
    
    # Test all agents in parallel
    (curl -X POST "$MANAGER_URL" -H "Content-Type: application/json" \
        -d '{"action": "health_check", "load_test": true}' -o /dev/null -s) &
    
    (curl -X POST "$RESEARCHER_URL" -H "Content-Type: application/json" \
        -d '{"type": "performance_metrics", "load_test": true}' -o /dev/null -s) &
    
    (curl -X POST "$DATA_PROCESSOR_URL" -H "Content-Type: application/json" \
        -d '{"data_type": "time_series", "load_test": true}' -o /dev/null -s) &
    
    sleep 30  # Wait 30 seconds between rounds
done

wait
echo "Load test complete!"
EOF

chmod +x load_test.sh

# Threshold adjustment script
cat > adjust_thresholds.sh << 'EOF'
#!/bin/bash
# Adjust CloudWatch alarm thresholds based on performance

source .env

echo "Adjusting alarm thresholds..."
echo "============================="

# Function to update alarm threshold
update_alarm() {
    local alarm_name=$1
    local new_threshold=$2
    
    aws cloudwatch put-metric-alarm \
        --alarm-name "$alarm_name" \
        --threshold "$new_threshold" \
        --output text
    
    echo "Updated $alarm_name to threshold: $new_threshold"
}

# Adjust based on your observations
# Example adjustments (modify as needed):

# If getting too many false positives on errors:
# update_alarm "cabruca-segmentation-prod-manager-agent-errors" 10

# If Lambda duration alarms are too sensitive:
# update_alarm "cabruca-segmentation-prod-manager-agent-duration" 240000

# If throttling is expected during peaks:
# update_alarm "cabruca-segmentation-prod-data-processor-agent-throttles" 5

echo "Threshold adjustments complete!"
EOF

chmod +x adjust_thresholds.sh

echo ""
echo -e "${GREEN}✅ Helper scripts created:${NC}"
echo "  - monitor_costs.sh    : Daily cost monitoring"
echo "  - load_test.sh        : Gradual load testing"
echo "  - adjust_thresholds.sh: Alarm threshold tuning"
echo ""

# Final summary
echo "========================================="
echo -e "${GREEN}🎉 Setup Complete!${NC}"
echo "========================================="
echo ""
echo -e "${CYAN}System Status:${NC}"
cd terraform > /dev/null 2>&1
if terraform output > /dev/null 2>&1; then
    echo "  ✅ Infrastructure: Deployed"
    echo "  ✅ Agents: $(terraform output -json agent_lambda_functions 2>/dev/null | jq -r 'keys | length') deployed"
    echo "  ✅ DynamoDB: Tables created"
    echo "  ✅ S3: Buckets ready"
    echo "  ✅ Monitoring: CloudWatch active"
else
    echo "  ⚠️  Infrastructure not yet deployed"
fi
cd .. > /dev/null 2>&1

echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "1. Monitor costs: ./monitor_costs.sh"
echo "2. Run load test: ./load_test.sh"
echo "3. Check AgentOps: https://app.agentops.ai"
echo "4. Review CloudWatch: $DASHBOARD_URL"
echo "5. Adjust thresholds: ./adjust_thresholds.sh"
echo ""
echo -e "${YELLOW}Support:${NC}"
echo "  - Runbook: terraform/INCIDENT_RESPONSE_RUNBOOK.md"
echo "  - Deploy Guide: terraform/DEPLOYMENT_GUIDE.md"
echo "  - Test Suite: terraform/test_agents.sh"
echo ""
echo "Happy monitoring! 🎯"