# QA Engineer Agent

Automated testing agent for validating deployments and changes in the Cabruca Segmentation system.

## Overview

The QA Engineer Agent is a serverless Lambda function that automatically validates deployments through comprehensive testing:

- **Cypress E2E Tests**: Automatically generates and runs end-to-end tests
- **API Validation**: Tests API endpoints using FastAPI test client approach
- **Cost Compliance**: Ensures changes don't exceed budget thresholds
- **Test Reporting**: Reports results back via EventBridge

## Features

### 1. Automated Test Generation
- Generates Cypress test scripts based on deployment changes
- Creates API tests for modified endpoints
- Produces UI component tests for frontend changes
- Generates performance tests for all deployments

### 2. Test Execution
- Triggers CodeBuild projects to run Cypress tests
- Validates API endpoints with health checks
- Performs response time validation
- Checks JSON structure and status codes

### 3. Cost Compliance
- Analyzes current monthly AWS costs
- Estimates cost impact of changes
- Checks against configured thresholds ($100/month default)
- Generates cost optimization recommendations

### 4. Test Result Management
- Stores results in DynamoDB
- Uploads test artifacts to S3
- Publishes events to EventBridge
- Maintains test history and metrics

## Architecture

```
EventBridge (Deployment Events)
    ↓
QA Agent Lambda (512MB, 5min timeout)
    ↓
    ├── Generate Cypress Tests → S3
    ├── Trigger CodeBuild → Run Tests
    ├── Validate API Endpoints
    ├── Check Cost Compliance
    └── Report Results → EventBridge
```

## Configuration

### Environment Variables
- `TEST_RESULTS_TABLE`: DynamoDB table for test results
- `CYPRESS_PROJECT_NAME`: CodeBuild project name
- `API_ENDPOINT`: API URL for validation
- `S3_BUCKET`: Bucket for test artifacts
- `EVENT_BUS_NAME`: EventBridge bus name
- `COST_THRESHOLD`: Monthly cost limit (USD)
- `ENVIRONMENT`: Deployment environment

### Terraform Variables
```hcl
variable "api_endpoint" {
  description = "API endpoint URL for QA testing"
  type        = string
  default     = ""
}

variable "cost_threshold" {
  description = "Monthly cost threshold in USD"
  type        = number
  default     = 100
}

variable "eventbridge_bus_name" {
  description = "EventBridge bus for agent communication"
  type        = string
  default     = ""
}
```

## Deployment

1. **Package the Lambda function**:
```bash
cd qa_agent
pip install -r requirements.txt -t .
cd ..
```

2. **Deploy with Terraform**:
```bash
terraform plan
terraform apply
```

3. **Verify deployment**:
```bash
aws lambda get-function --function-name <environment>-qa-agent
```

## Testing

### Local Testing
```python
# Test the Lambda handler locally
python -c "
import json
from lambda_function import lambda_handler

with open('test_event.json', 'r') as f:
    event = json.load(f)
    
result = lambda_handler(event, {})
print(json.dumps(result, indent=2))
"
```

### Manual Event Testing
```bash
# Send test event to Lambda
aws lambda invoke \
  --function-name <environment>-qa-agent \
  --payload file://test_event.json \
  response.json
```

## Test Types

### Cypress Tests Generated

1. **API Endpoint Tests**
   - Validates response status
   - Checks response structure
   - Tests error handling

2. **UI Component Tests**
   - Verifies component visibility
   - Tests user interactions
   - Validates state changes

3. **Form Validation Tests**
   - Tests required field validation
   - Validates form submission
   - Checks error messages

4. **Performance Tests**
   - Measures page load time
   - Validates against 3-second threshold
   - Records performance metrics

## Cost Analysis

### Monthly Cost Breakdown
- Lambda Invocations: ~$2/month (assuming 1000 tests)
- CodeBuild Runs: ~$2/month (100 builds × 5 min)
- DynamoDB: ~$0.50/month (PAY_PER_REQUEST)
- S3 Storage: ~$0.50/month (test artifacts)
- **Total: ~$5/month**

### Cost Optimization
- Uses minimum Lambda memory (512MB)
- Implements S3 lifecycle policies (30-day retention)
- CodeBuild uses smallest instance type
- DynamoDB uses on-demand pricing

## Monitoring

### CloudWatch Metrics
- Lambda invocations and errors
- CodeBuild success rate
- Test execution duration
- Cost compliance status

### Dashboard Widgets
- QA Agent performance metrics
- Cypress test execution stats
- Test pass/fail rates
- Cost trend analysis

## Event Flow

1. **Input**: Deployment completion event from Engineer Agent
2. **Processing**:
   - Parse deployment details
   - Generate test scripts
   - Execute tests in parallel
   - Analyze costs
3. **Output**: Test results event with:
   - Test suite summary
   - Individual test results
   - Cost compliance status
   - Recommendations

## Troubleshooting

### Common Issues

1. **CodeBuild Timeout**
   - Check Cypress test complexity
   - Verify network connectivity
   - Review build logs in CloudWatch

2. **Cost Analysis Failures**
   - Ensure Cost Explorer API access
   - Check IAM permissions for ce:GetCostAndUsage
   - Verify date range calculations

3. **S3 Upload Errors**
   - Check bucket permissions
   - Verify bucket exists
   - Review S3 bucket policies

### Debug Logs
```bash
# View Lambda logs
aws logs tail /aws/lambda/<environment>-qa-agent --follow

# View CodeBuild logs
aws logs tail /aws/codebuild/<environment>-cypress-tests --follow
```

## Integration Points

### Listens To
- `engineer.agent`: Deployment completion events

### Publishes To
- `qa.agent`: Test result events

### AWS Services Used
- Lambda (function execution)
- DynamoDB (test results storage)
- S3 (test artifacts)
- CodeBuild (Cypress execution)
- Cost Explorer (cost analysis)
- EventBridge (event routing)
- CloudWatch (monitoring)

## Best Practices

1. **Test Generation**
   - Keep tests focused and atomic
   - Use data-testid attributes for selectors
   - Generate performance tests for all deployments

2. **Cost Management**
   - Set appropriate thresholds
   - Review cost recommendations
   - Monitor trend analysis

3. **Test Maintenance**
   - Clean up old test artifacts
   - Review failing tests regularly
   - Update test templates as needed

## Future Enhancements

- [ ] Support for Selenium tests
- [ ] Integration with SonarQube
- [ ] Visual regression testing
- [ ] Load testing capabilities
- [ ] Security scanning integration
- [ ] Mobile app testing support
