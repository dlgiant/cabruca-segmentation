# Manager Agent Lambda Function

## Overview

The Manager Agent is an intelligent monitoring system that runs as an AWS Lambda function. It monitors system health, analyzes user feedback, makes intelligent decisions using LangChain with Claude Opus-4, and publishes events to EventBridge when issues or opportunities are identified.

## Features

### 🔍 System Monitoring
- **CloudWatch Metrics Monitoring**
  - API Gateway latency tracking
  - Lambda function error rates
  - Lambda invocation counts
  - Automatic error rate calculation

### 💬 User Feedback Analysis
- Analyzes feedback from DynamoDB table
- Calculates sentiment scores
- Tracks negative feedback patterns
- Identifies user experience issues

### 🤖 Intelligent Decision Making
- Uses LangChain with Claude Opus-4 for analysis
- Provides root cause analysis
- Recommends specific actions
- Determines severity levels
- Suggests auto-remediation when applicable

### 💰 Cost Tracking
- Monitors AWS costs using Cost Explorer API
- Compares current vs previous month costs
- Identifies cost anomalies
- Tracks costs by service

### 🚨 Event Publishing
- Publishes critical issues to EventBridge
- Sends opportunity events for optimization
- Structured event format for downstream processing

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     EventBridge (Scheduler)                  │
│                     Triggers every 30 minutes                │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                    Manager Agent Lambda                      │
│                     (512MB, 5min timeout)                    │
├─────────────────────────────────────────────────────────────┤
│  • Collects CloudWatch metrics                               │
│  • Analyzes DynamoDB feedback                                │
│  • Tracks AWS costs                                          │
│  • Uses LangChain + Claude for decisions                     │
│  • Publishes events to EventBridge                           │
└─────────────────────────────────────────────────────────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │CloudWatch│   │ DynamoDB │   │   Cost   │
        │ Metrics  │   │ Feedback │   │ Explorer │
        └──────────┘   └──────────┘   └──────────┘
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `FEEDBACK_TABLE_NAME` | DynamoDB table name for user feedback | `user-feedback` |
| `EVENT_BUS_NAME` | EventBridge bus name | `default` |
| `ANTHROPIC_API_KEY` | API key for Claude access | Required |
| `ENVIRONMENT` | Environment name (dev/staging/production) | `production` |

### Thresholds

| Metric | Threshold | Description |
|--------|-----------|-------------|
| Error Rate | 5% | Triggers high error rate issue |
| API Latency | 3000ms | Triggers high latency issue |
| Negative Feedback | 20% | Triggers negative feedback issue |
| Cost Increase | 15% | Triggers cost anomaly issue |

## Deployment

### Prerequisites

1. **Python 3.11** installed locally
2. **AWS CLI** configured with appropriate credentials
3. **Terraform** >= 1.0 installed
4. **Anthropic API Key** for Claude access

### Quick Deploy

1. **Package the Lambda function:**
   ```bash
   cd manager_agent
   ./deploy.sh
   ```

2. **Configure Terraform:**
   ```bash
   cd terraform
   terraform init
   ```

3. **Create terraform.tfvars:**
   ```hcl
   anthropic_api_key = "your-anthropic-api-key"
   environment = "production"
   feedback_table_name = "user-feedback"
   ```

4. **Deploy:**
   ```bash
   terraform plan
   terraform apply
   ```

## Event Schema

### System Issue Event
```json
{
  "Source": "manager.agent",
  "DetailType": "SystemIssue.<issue_type>",
  "Detail": {
    "issue": {
      "type": "high_error_rate",
      "severity": "high",
      "description": "Error rate (7.5%) exceeds threshold (5%)",
      "affected_service": "Lambda Functions",
      "metrics": {
        "error_rate": 0.075
      },
      "recommended_action": "Review recent deployments",
      "timestamp": "2024-01-01T12:00:00Z"
    },
    "llm_analysis": {
      "severity": "high",
      "should_alert": true,
      "recommended_actions": [...],
      "root_cause_analysis": "...",
      "auto_remediation": "..."
    },
    "environment": "production"
  }
}
```

### Opportunity Event
```json
{
  "Source": "manager.agent",
  "DetailType": "Opportunity.AutoRemediation",
  "Detail": {
    "action": "Scale Lambda concurrent executions",
    "recommendations": [
      "Increase reserved concurrency to 100",
      "Enable auto-scaling for API Gateway"
    ],
    "context": {
      "metrics_summary": {
        "error_rate": 0.02,
        "avg_latency": 2500
      }
    }
  }
}
```

## Issue Types

| Type | Description | Severity |
|------|-------------|----------|
| `HIGH_ERROR_RATE` | Lambda error rate exceeds threshold | HIGH/CRITICAL |
| `HIGH_LATENCY` | API latency exceeds threshold | MEDIUM |
| `NEGATIVE_FEEDBACK` | High ratio of negative user feedback | HIGH |
| `COST_ANOMALY` | Significant cost increase detected | MEDIUM |
| `PERFORMANCE_DEGRADATION` | General performance issues | MEDIUM |

## Cost Breakdown

### Estimated Monthly Costs (~$5/month)

| Component | Usage | Cost |
|-----------|-------|------|
| Lambda Invocations | 1,440/month (48/day) | $0.29 |
| Lambda Duration | ~7,200 seconds @ 512MB | $0.06 |
| CloudWatch Logs | ~100MB | $0.05 |
| CloudWatch Metrics API | ~1,440 requests | $0.01 |
| DynamoDB Reads | ~1,440 scans | $0.18 |
| Cost Explorer API | ~1,440 calls | $0.01 |
| EventBridge Events | ~100 events | $0.00 |
| **Anthropic API** | ~1,440 calls | ~$4.32 |
| **Total** | | **~$5/month** |

## Monitoring

### CloudWatch Metrics
- **Invocations**: Number of times the function runs
- **Duration**: Execution time per invocation
- **Errors**: Number of failed executions
- **Throttles**: Rate limiting occurrences

### CloudWatch Logs
- All logs are stored in `/aws/lambda/manager-agent-{environment}`
- 7-day retention policy by default
- Structured JSON logging for easy querying

### Alarms
- **High Error Rate**: Triggers when >5 errors in 10 minutes
- Can be extended with SNS notifications

## Testing

### Local Testing
```python
# Test the Lambda function locally
import json
from lambda_function import lambda_handler

# Mock event
event = {
    "source": "manual",
    "detail": {"trigger": "test"}
}

# Mock context
class Context:
    function_name = "test-function"
    request_id = "test-request-id"

result = lambda_handler(event, Context())
print(json.dumps(result, indent=2))
```

### Integration Testing
1. Deploy to a test environment
2. Manually trigger the Lambda
3. Verify CloudWatch logs
4. Check EventBridge for published events

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Lambda timeout | Increase timeout in Terraform (max 15 minutes) |
| Missing permissions | Check IAM policy in Terraform configuration |
| High costs | Adjust scheduling frequency or reduce CloudWatch API calls |
| No LLM analysis | Verify ANTHROPIC_API_KEY is set correctly |
| DynamoDB errors | Ensure feedback table exists and has proper schema |

### Debug Mode
Set log level to DEBUG in the Lambda function:
```python
logger.setLevel(logging.DEBUG)
```

## Security Considerations

1. **API Key Management**: Store Anthropic API key in AWS Secrets Manager for production
2. **IAM Permissions**: Follow least privilege principle
3. **Data Privacy**: Ensure feedback data doesn't contain PII
4. **Network Security**: Lambda runs in VPC if needed for private resources

## Extension Points

### Adding New Metrics
1. Update `collect_cloudwatch_metrics()` method
2. Add new threshold constants
3. Update `detect_issues()` method

### Adding New Issue Types
1. Add to `IssueType` enum
2. Implement detection logic in `detect_issues()`
3. Update event publishing logic

### Customizing LLM Analysis
1. Modify prompt template in `analyze_with_llm()`
2. Update `AgentDecision` model
3. Adjust temperature and max_tokens

## Support

For issues or questions:
1. Check CloudWatch Logs for error details
2. Review EventBridge for published events
3. Verify all AWS services are accessible
4. Ensure sufficient IAM permissions

## License

This implementation is provided as-is for the autonomous agent system architecture.
