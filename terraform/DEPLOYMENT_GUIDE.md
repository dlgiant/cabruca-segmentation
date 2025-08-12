# 🚀 Multi-Agent System Deployment Guide

## Prerequisites

### Required Tools
- AWS CLI configured with credentials
- Terraform >= 1.0
- Python 3.11
- jq (for JSON parsing)
- zip (for Lambda packaging)

### Required AWS Permissions
- Lambda: Full access
- DynamoDB: Full access
- S3: Full access
- CloudWatch: Full access
- IAM: Create roles and policies
- EventBridge: Full access
- VPC: Full access (if not using existing)

### Environment Setup
```bash
# Set your AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="sa-east-1"

# Set AgentOps API key (get from https://app.agentops.ai)
export AGENTOPS_API_KEY="your-agentops-api-key"

# Optional: Set alert email
export ALERT_EMAIL="your-email@example.com"
```

---

## 📋 Step-by-Step Deployment

### Step 1: Prepare the Environment

```bash
# Clone the repository (if not already done)
cd cabruca-segmentation/terraform

# Make scripts executable
chmod +x deploy_agents.sh
chmod +x test_agents.sh
chmod +x agent_env_config.sh
chmod +x package_lambda.sh

# Install Python dependencies for Lambda packaging
pip install -r requirements.txt
```

### Step 2: Configure Variables

Edit `mvp.tfvars` to customize your deployment:

```hcl
# Key variables to update:
alert_email = "your-email@example.com"  # For cost and error alerts
monitoring_configuration = {
  alarm_email = "your-email@example.com"
  slack_webhook_url = "https://hooks.slack.com/services/YOUR/WEBHOOK/URL"
}
```

### Step 3: Deploy Infrastructure

```bash
# Run the deployment script
./deploy_agents.sh

# The script will:
# 1. Package all Lambda functions
# 2. Initialize Terraform
# 3. Validate configuration
# 4. Create a deployment plan
# 5. Apply the infrastructure (requires confirmation)
```

### Step 4: Configure Environment Variables

After deployment, configure the Lambda environment variables:

```bash
# Set the AgentOps API key
export AGENTOPS_API_KEY="your-actual-api-key"

# Run the configuration script
./agent_env_config.sh prod
```

### Step 5: Verify Deployment

```bash
# Run the comprehensive test suite
./test_agents.sh

# Check individual components:
# - Lambda functions deployed
# - DynamoDB tables created
# - S3 buckets available
# - CloudWatch logs active
# - EventBridge rules configured
```

### Step 6: Setup Monitoring

1. **CloudWatch Dashboard**:
   - Navigate to the URL shown in deployment output
   - Customize widgets as needed
   - Set up custom metrics

2. **AgentOps Dashboard**:
   ```bash
   # Login to AgentOps
   open https://app.agentops.ai
   
   # Configure:
   # - Create new project: "Cabruca Segmentation"
   # - Add agents with matching names
   # - Set up custom events and metrics
   # - Configure alerting rules
   ```

3. **Cost Monitoring**:
   ```bash
   # Set up AWS Budget
   aws budgets create-budget \
     --account-id $(aws sts get-caller-identity --query Account --output text) \
     --budget file://budget-config.json \
     --notifications-with-subscribers file://budget-notifications.json
   ```

---

## 🧪 Testing the System

### Test 1: Manager Agent Orchestration
```bash
# Test metric anomaly detection
curl -X POST $(terraform output -raw manager_agent_url) \
  -H "Content-Type: application/json" \
  -d '{
    "action": "analyze_metrics",
    "metrics": {
      "cpu_usage": 95,
      "memory_usage": 88,
      "error_rate": 15
    }
  }'
```

### Test 2: Data Processing Pipeline
```bash
# Upload test data to trigger processing
aws s3 cp test-data/sample-image.tif \
  s3://$(terraform output -raw agent_queue_bucket)/input/

# Check processing results
aws s3 ls s3://$(terraform output -raw agent_artifacts_bucket)/processed/
```

### Test 3: Research Analysis
```bash
# Request cabruca analysis
curl -X POST $(terraform output -raw researcher_agent_url) \
  -H "Content-Type: application/json" \
  -d '{
    "type": "cabruca_analysis",
    "region": "bahia",
    "parameters": {
      "include_recommendations": true
    }
  }'
```

---

## 📊 First 24-Hour Monitoring

### Hour 1-4: Initial Validation
- ✅ All Lambda functions responding
- ✅ No error alarms triggered
- ✅ DynamoDB tables receiving data
- ✅ S3 buckets accessible
- ✅ CloudWatch logs streaming

### Hour 4-8: Load Testing
```bash
# Run gradual load test
for i in {1..100}; do
  ./test_agents.sh &
  sleep 60  # 1 request per minute
done
```

### Hour 8-12: Performance Tuning
- Adjust Lambda memory if needed
- Tune DynamoDB capacity
- Review CloudWatch metrics
- Optimize S3 lifecycle policies

### Hour 12-24: Production Readiness
- Monitor cost trends
- Review error patterns
- Adjust alarm thresholds
- Document any issues

---

## ⚙️ Threshold Adjustments

Based on initial performance, adjust these thresholds:

```bash
# Update Lambda timeouts if needed
aws lambda update-function-configuration \
  --function-name cabruca-segmentation-prod-manager-agent \
  --timeout 600  # Increase to 10 minutes if needed

# Adjust CloudWatch alarms
aws cloudwatch put-metric-alarm \
  --alarm-name cabruca-segmentation-prod-manager-agent-errors \
  --threshold 10  # Increase if false positives

# Update DynamoDB if throttling occurs
aws dynamodb update-table \
  --table-name cabruca-segmentation-prod-agent-state \
  --billing-mode PROVISIONED \
  --provisioned-throughput ReadCapacityUnits=10,WriteCapacityUnits=10
```

---

## 🛠️ Maintenance Tasks

### Daily
```bash
# Check system health
./test_agents.sh

# Review costs
aws ce get-cost-and-usage \
  --time-period Start=$(date -d yesterday +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE | jq '.ResultsByTime[0].Total.UnblendedCost.Amount'
```

### Weekly
```bash
# Clean up old data
aws s3 rm s3://$(terraform output -raw agent_artifacts_bucket)/processed/ \
  --recursive \
  --exclude "*" \
  --include "*.json" \
  --older-than 7

# Review and rotate logs
aws logs put-retention-policy \
  --log-group-name /aws/lambda/cabruca-segmentation-prod-manager-agent \
  --retention-in-days 14
```

### Monthly
```bash
# Full system backup
./backup_system.sh

# Update dependencies
cd terraform
for agent in */; do
  cd $agent
  pip install -r requirements.txt --upgrade -t package/
  cd ..
done

# Review and optimize costs
aws ce get-cost-forecast \
  --time-period Start=$(date +%Y-%m-%d),End=$(date -d '+1 month' +%Y-%m-%d) \
  --metric UNBLENDED_COST \
  --granularity MONTHLY
```

---

## 🔧 Troubleshooting

### Common Issues and Solutions

#### Lambda Timeout Errors
```bash
# Increase timeout
aws lambda update-function-configuration \
  --function-name cabruca-segmentation-prod-${AGENT}-agent \
  --timeout 900
```

#### DynamoDB Throttling
```bash
# Switch to on-demand
aws dynamodb update-table \
  --table-name cabruca-segmentation-prod-agent-${TABLE} \
  --billing-mode PAY_PER_REQUEST
```

#### High Costs
```bash
# Emergency shutdown
terraform apply -var="api_min_instances=0" -var="api_max_instances=0" -auto-approve

# Review cost breakdown
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=LINKED_ACCOUNT \
  --group-by Type=DIMENSION,Key=SERVICE
```

---

## 📚 Additional Resources

- [AWS Lambda Best Practices](https://docs.aws.amazon.com/lambda/latest/dg/best-practices.html)
- [DynamoDB Performance Guide](https://docs.aws.amazon.com/amazondynamodb/latest/developerguide/bp-general.html)
- [AgentOps Documentation](https://docs.agentops.ai)
- [Terraform AWS Provider](https://registry.terraform.io/providers/hashicorp/aws/latest)

---

## 🚨 Emergency Contacts

- **AWS Support**: [Console](https://console.aws.amazon.com/support)
- **AgentOps Support**: support@agentops.ai
- **On-Call Engineer**: [PagerDuty](https://your-domain.pagerduty.com)

---

**Version**: 1.0.0
**Last Updated**: March 2024
**Next Review**: Monthly