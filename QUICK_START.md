# 🚀 Quick Start Guide - Multi-Agent System Deployment

This guide will help you deploy the complete multi-agent system in just a few steps.

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **AWS CLI** installed and configured
3. **Terraform** >= 1.0 installed
4. **Python** 3.11 installed
5. **AgentOps Account** (free at https://app.agentops.ai)

## 🎯 5-Step Deployment Process

### Step 1: Clone and Setup Environment

```bash
# Navigate to the project directory
cd cabruca-segmentation

# Copy the environment template
cp .env.example .env
```

### Step 2: Configure Environment Variables

Edit the `.env` file with your actual values:

```bash
# Edit with your preferred editor
nano .env  # or: code .env
```

**Required variables to update:**

```env
# AWS Configuration (REQUIRED)
AWS_ACCESS_KEY_ID=your_actual_key_here
AWS_SECRET_ACCESS_KEY=your_actual_secret_here
AWS_ACCOUNT_ID=123456789012  # Get with: aws sts get-caller-identity

# AgentOps Configuration (REQUIRED)
AGENTOPS_API_KEY=your_agentops_key_here  # From https://app.agentops.ai

# Alert Configuration (REQUIRED)
ALERT_EMAIL=your-email@example.com
ALERT_PHONE=+55-11-98765-4321  # Optional but recommended

# Security Keys (REQUIRED - generate secure values)
API_KEY=generate_a_secure_api_key_here
SECRET_KEY=generate_a_secure_secret_key_here
```

### Step 3: Run Automated Setup

```bash
# Run the complete setup script
./setup_deployment.sh
```

This script will:
- ✅ Validate your environment variables
- ✅ Deploy all AWS infrastructure
- ✅ Configure Lambda functions
- ✅ Run comprehensive tests
- ✅ Set up monitoring
- ✅ Create helper scripts

### Step 4: Configure AgentOps Dashboard

1. **Login to AgentOps**: https://app.agentops.ai
2. **Create New Project**: "Cabruca Segmentation"
3. **Add Your Agents**:
   - manager-agent
   - engineer-agent
   - qa-agent
   - researcher-agent
   - data-processor-agent
4. **Configure Alerts**:
   - Set up email notifications
   - Configure error thresholds
   - Enable session tracking

### Step 5: Monitor for 24 Hours

Use the provided monitoring scripts:

```bash
# Check costs (run daily)
./monitor_costs.sh

# Run load test (gradual increase)
./load_test.sh

# Monitor in real-time
cd terraform
./test_agents.sh
```

## 📊 24-Hour Monitoring Checklist

### ⏰ Hour 1-4: Initial Validation
```bash
# Test all agents
cd terraform && ./test_agents.sh

# Check CloudWatch logs
aws logs tail /aws/lambda/cabruca-segmentation-prod-manager-agent --follow
```

✅ Checklist:
- [ ] All Lambda functions responding
- [ ] No critical alarms triggered
- [ ] DynamoDB tables have data
- [ ] S3 buckets accessible
- [ ] AgentOps showing sessions

### ⏰ Hour 4-8: Load Testing
```bash
# Run gradual load test
./load_test.sh

# Monitor metrics
watch -n 60 './monitor_costs.sh'
```

✅ Checklist:
- [ ] No throttling errors
- [ ] Response times < 1 second
- [ ] Memory usage < 80%
- [ ] No Lambda timeouts

### ⏰ Hour 8-12: Performance Tuning
```bash
# Adjust if needed
cd terraform

# Increase memory for slow agents
aws lambda update-function-configuration \
  --function-name cabruca-segmentation-prod-manager-agent \
  --memory-size 2048

# Update alarm thresholds
./adjust_thresholds.sh
```

✅ Checklist:
- [ ] Optimize Lambda memory
- [ ] Adjust DynamoDB capacity
- [ ] Review error patterns
- [ ] Update alarm thresholds

### ⏰ Hour 12-24: Production Readiness
```bash
# Final validation
cd terraform && ./test_agents.sh

# Cost review
./monitor_costs.sh

# Document issues
echo "Issues found:" >> monitoring_notes.txt
```

✅ Checklist:
- [ ] Cost within budget ($10-35/month)
- [ ] All tests passing
- [ ] Documentation updated
- [ ] Team notified

## 🛠️ Useful Commands

### Test Individual Agents
```bash
# Test Manager Agent
curl -X POST $(cd terraform && terraform output -raw agent_lambda_functions | jq -r '.manager.invoke_url') \
  -H "Content-Type: application/json" \
  -d '{"action": "health_check"}'

# Test Researcher Agent
curl -X POST $(cd terraform && terraform output -raw agent_lambda_functions | jq -r '.researcher.invoke_url') \
  -H "Content-Type: application/json" \
  -d '{"type": "cabruca_analysis", "region": "bahia"}'
```

### Check System Status
```bash
# Get all Lambda function states
for agent in manager engineer qa researcher data_processor; do
  echo "$agent: $(aws lambda get-function --function-name cabruca-segmentation-prod-${agent}-agent --query 'Configuration.State' --output text)"
done

# Check DynamoDB item counts
for table in state memory tasks; do
  echo "$table: $(aws dynamodb scan --table-name cabruca-segmentation-prod-agent-$table --select COUNT --query Count --output text) items"
done
```

### Emergency Procedures
```bash
# Stop all agents (emergency)
for agent in manager engineer qa researcher data_processor; do
  aws lambda put-function-concurrency \
    --function-name cabruca-segmentation-prod-${agent}-agent \
    --reserved-concurrent-executions 0
done

# Destroy everything (careful!)
cd terraform
terraform destroy -var-file=mvp.tfvars -auto-approve
```

## 📈 Cost Optimization Tips

1. **Use Scheduled Shutdown** (already configured for dev):
   - Agents auto-shutdown at 10 PM on weekdays
   - Saves ~40% on Lambda costs

2. **Monitor DynamoDB Usage**:
   ```bash
   # Switch to on-demand if low usage
   aws dynamodb update-table \
     --table-name cabruca-segmentation-prod-agent-state \
     --billing-mode PAY_PER_REQUEST
   ```

3. **Clean Up Old Data**:
   ```bash
   # Remove old S3 artifacts (> 7 days)
   aws s3 rm s3://cabruca-segmentation-prod-agent-artifacts-${AWS_ACCOUNT_ID}/ \
     --recursive --exclude "*" --include "*.json" \
     --older-than 7
   ```

## 🔧 Troubleshooting

### Common Issues

**Issue: Lambda timeout errors**
```bash
# Increase timeout to 10 minutes
aws lambda update-function-configuration \
  --function-name cabruca-segmentation-prod-${AGENT}-agent \
  --timeout 600
```

**Issue: AgentOps not showing data**
```bash
# Verify API key is set
aws lambda get-function-configuration \
  --function-name cabruca-segmentation-prod-manager-agent \
  --query 'Environment.Variables.AGENTOPS_API_KEY'
```

**Issue: High costs**
```bash
# Check cost breakdown
aws ce get-cost-and-usage \
  --time-period Start=$(date -d '7 days ago' +%Y-%m-%d),End=$(date +%Y-%m-%d) \
  --granularity DAILY \
  --metrics UnblendedCost \
  --group-by Type=DIMENSION,Key=SERVICE
```

## 📚 Documentation

- **Deployment Guide**: `terraform/DEPLOYMENT_GUIDE.md`
- **Incident Response**: `terraform/INCIDENT_RESPONSE_RUNBOOK.md`
- **Agent Documentation**: `terraform/*/README.md`
- **API Documentation**: Available after deployment at ALB URL

## 🆘 Support

- **AWS Issues**: Check CloudWatch Logs first
- **AgentOps Issues**: support@agentops.ai
- **Cost Concerns**: Review `monitor_costs.sh` output
- **Emergency**: Follow `INCIDENT_RESPONSE_RUNBOOK.md`

## ✅ Success Criteria

Your deployment is successful when:
1. ✅ All 5 agents pass health checks
2. ✅ AgentOps shows active sessions
3. ✅ No critical CloudWatch alarms
4. ✅ Costs projecting < $35/month
5. ✅ All test suite passes

---

**Estimated Time**: 30-45 minutes for initial deployment
**Monthly Cost**: $10-35 USD (MVP configuration)
**Support**: Create an issue in the repository for help