# Cost Control Infrastructure

## Overview

This module implements comprehensive cost control mechanisms to ensure AWS spending stays under $500/month. It includes automatic safeguards, monitoring, alerting, and circuit breakers to prevent cost overruns.

## Features Implemented

### 1. Lambda Concurrency Limits
- **Manager Agent**: 1 concurrent execution
- **Engineer Agent**: 2 concurrent executions  
- **QA Agent**: 2 concurrent executions

These limits prevent runaway Lambda invocations that could spike costs.

### 2. API Gateway Request Throttling
- **Rate Limit**: 100 requests per minute
- **Burst Limit**: 200 requests
- **Hourly Quota**: 6,000 requests

Prevents excessive API calls that could trigger unnecessary Lambda executions.

### 3. Cost Allocation Tags
All resources are tagged with:
- `CostCenter`: engineering, qa, operations, monitoring
- `Agent`: manager, engineer, qa, monitoring
- `Environment`: development, staging, production

This enables detailed cost tracking and attribution.

### 4. Auto-Shutdown for Development
Development resources automatically:
- **Shutdown**: 8 PM BRT (weekdays)
- **Startup**: 7 AM BRT (weekdays)
- **Resources affected**: Lambda functions, EC2 instances, RDS databases

Saves ~60% of development costs by shutting down during non-working hours.

### 5. CloudWatch Alarms
Three-tier alerting system:
- **80% threshold** ($400): Warning alert via email
- **90% threshold** ($450): Critical alert via email + SMS
- **100% threshold** ($500): Emergency alert + automatic circuit breaker

### 6. Cost Reports Dashboard
Real-time dashboard showing:
- Monthly cost tracking with budget visualization
- Agent invocation metrics
- Execution duration trends
- Concurrent execution monitoring
- API Gateway metrics
- Cost breakdown by service

Access the dashboard at:
```
https://console.aws.amazon.com/cloudwatch/home?region=sa-east-1#dashboards:name=production-cost-control-dashboard
```

### 7. Circuit Breaker System
Automatic cost protection that:
- Monitors costs every 10 minutes
- Triggers when projected monthly costs exceed $500
- Halts all agent Lambda functions (sets concurrency to 0)
- Disables EventBridge rules
- Sends critical alerts
- Auto-recovers when costs drop below 70% threshold

## Setup Instructions

### 1. Prerequisites
- AWS Account with appropriate permissions
- Terraform >= 1.0
- AWS CLI configured
- Cost Explorer API enabled in your AWS account

### 2. Configuration

Create a `terraform.tfvars` file:

```hcl
# Required variables
environment = "production"
alert_email = "your-email@example.com"
alert_phone = "+5511999999999"  # Optional, for SMS alerts

# API Keys
anthropic_api_key = "sk-ant-api03-..."
github_repo = "your-org/your-repo"

# Optional overrides
cost_alert_threshold = 400  # Default: $400 (80% of $500)
enable_org_policies = false  # Set true if using AWS Organizations
```

### 3. Deployment

```bash
# Initialize Terraform
terraform init

# Review the plan
terraform plan

# Apply the configuration
terraform apply

# Note the outputs
terraform output
```

### 4. Post-Deployment

1. **Subscribe to SNS Topics**: Check your email and confirm the SNS subscriptions
2. **Test Circuit Breaker**: Run the circuit breaker Lambda manually to verify it works
3. **Configure AWS Budgets**: The budget is created automatically but review settings
4. **Enable Cost Explorer**: Ensure Cost Explorer is enabled for detailed cost analysis

## Cost Breakdown

Estimated monthly costs with all safeguards:

| Component | Cost | Notes |
|-----------|------|-------|
| Lambda Functions | ~$50 | With concurrency limits |
| API Gateway | ~$10 | 100 req/min throttling |
| DynamoDB | ~$25 | On-demand billing |
| S3 Storage | ~$5 | Lifecycle policies enabled |
| CloudWatch | ~$10 | Logs, metrics, dashboards |
| SNS/EventBridge | ~$5 | Alerts and events |
| **Total** | **~$105** | Well under $500 limit |

## Monitoring

### Key Metrics to Watch

1. **Daily Cost Rate**
   - Normal: < $17/day
   - Warning: > $25/day
   - Critical: > $50/day

2. **Lambda Invocations**
   - Manager: < 50/day
   - Engineer: < 100/day
   - QA: < 200/day

3. **API Gateway Requests**
   - Should stay under 6,000/hour
   - Monitor 4XX/5XX error rates

### Alert Response Playbook

#### 80% Budget Alert ($400)
1. Review CloudWatch dashboard for cost trends
2. Check for unusual Lambda invocation patterns
3. Verify no infinite loops in agent logic
4. Consider temporary reduction in agent activity

#### 90% Budget Alert ($450)
1. Immediately check circuit breaker status
2. Review Cost Explorer for service breakdown
3. Disable non-critical agent functions manually if needed
4. Investigate root cause of cost spike

#### Circuit Breaker Triggered
1. All agents are automatically halted
2. Review the alert message for cost breakdown
3. Check CloudWatch Logs for circuit breaker details
4. Fix the issue before re-enabling agents
5. Manually adjust Lambda concurrency to re-enable

## Manual Controls

### Disable All Agents
```bash
aws lambda put-function-concurrency \
  --function-name manager-agent-production \
  --reserved-concurrent-executions 0

aws lambda put-function-concurrency \
  --function-name engineer-agent-production \
  --reserved-concurrent-executions 0

aws lambda put-function-concurrency \
  --function-name production-qa-agent \
  --reserved-concurrent-executions 0
```

### Re-enable Agents
```bash
aws lambda put-function-concurrency \
  --function-name manager-agent-production \
  --reserved-concurrent-executions 1

aws lambda put-function-concurrency \
  --function-name engineer-agent-production \
  --reserved-concurrent-executions 2

aws lambda put-function-concurrency \
  --function-name production-qa-agent \
  --reserved-concurrent-executions 2
```

### Check Current Costs
```bash
# Get current month costs
aws ce get-cost-and-usage \
  --time-period Start=$(date +%Y-%m-01),End=$(date +%Y-%m-%d) \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --filter '{"Tags": {"Key": "CostCenter", "Values": ["engineering", "qa", "operations"]}}'
```

## Troubleshooting

### Circuit Breaker Not Triggering
1. Verify the Lambda function has proper IAM permissions
2. Check CloudWatch Events rule is enabled
3. Review Lambda function logs for errors
4. Ensure Cost Explorer API is enabled

### False Positive Triggers
1. Adjust `DAILY_LIMIT` environment variable
2. Review cost anomaly detection settings
3. Check for one-time charges (data transfer, etc.)

### Auto-Shutdown Not Working
1. Verify environment is set to "development"
2. Check EventBridge rules are enabled
3. Review Lambda function logs
4. Ensure resources have proper tags

## Best Practices

1. **Regular Reviews**: Check the cost dashboard weekly
2. **Tag Everything**: Ensure all resources have proper cost allocation tags
3. **Set Budgets**: Use AWS Budgets in addition to CloudWatch alarms
4. **Test Failsafes**: Regularly test circuit breaker and auto-shutdown
5. **Document Changes**: Update this README when modifying thresholds

## Support

For issues or questions:
1. Check CloudWatch Logs for detailed error messages
2. Review the cost control dashboard
3. Contact the DevOps team
4. Review AWS Cost Explorer for detailed analysis

## License

This infrastructure code is proprietary and confidential.
