# Engineer Agent Lambda Function

## Overview

The Engineer Agent is an autonomous AI-powered Lambda function that implements solutions based on events received from the Manager Agent. It uses LangChain's ReAct agent pattern with Claude Opus-4 to intelligently plan and execute code changes, create pull requests, and update infrastructure configurations.

## Features

- **Autonomous Implementation**: Uses LangChain ReAct agent pattern for intelligent planning and execution
- **Code Generation**: Leverages Claude Opus-4 for generating production-ready code
- **GitHub Integration**: Automatically creates branches, commits, and pull requests using PyGithub
- **Terraform Management**: Updates infrastructure configurations when needed
- **Event-Driven**: Triggered by EventBridge events from Manager Agent
- **Task Tracking**: Maintains task state in DynamoDB with full audit trail
- **Completion Events**: Publishes events for QA Agent to begin testing

## Architecture

```
Manager Agent → EventBridge → Engineer Agent → GitHub/Terraform
                                    ↓
                              QA Agent (via EventBridge)
```

## Prerequisites

1. **AWS Account** with appropriate permissions
2. **GitHub Account** with repository access
3. **Anthropic API Key** for Claude access
4. **Python 3.11** installed locally
5. **Terraform** >= 1.0 installed
6. **AWS CLI** configured

## Setup Instructions

### 1. Configure AWS Secrets

Store your sensitive credentials in AWS Secrets Manager:

```bash
# Store GitHub Personal Access Token
aws secretsmanager create-secret \
    --name github-token \
    --secret-string "ghp_your_github_token_here"

# Store Anthropic API Key
aws secretsmanager create-secret \
    --name anthropic-api-key \
    --secret-string "sk-ant-your_api_key_here"
```

### 2. Package the Lambda Function

Run the deployment script to create the Lambda packages:

```bash
chmod +x deploy.sh
./deploy.sh
```

This will create:
- `lambda_deployment.zip`: The Lambda function code
- `lambda_layer.zip`: Dependencies layer (LangChain, PyGithub, etc.)

### 3. Deploy with Terraform

Navigate to the terraform directory and deploy:

```bash
cd terraform/
terraform init
terraform plan -var="github_repo=your-org/your-repo"
terraform apply -var="github_repo=your-org/your-repo"
```

### 4. Verify Deployment

Check that the Lambda function is created:

```bash
aws lambda get-function --function-name engineer-agent-production
```

## Configuration

### Environment Variables

The Lambda function uses the following environment variables (configured via Terraform):

| Variable | Description | Default |
|----------|-------------|---------|
| `GITHUB_TOKEN_SECRET_NAME` | Secrets Manager secret for GitHub token | `github-token` |
| `GITHUB_REPO` | Target GitHub repository (org/repo format) | Required |
| `ANTHROPIC_API_KEY_SECRET` | Secrets Manager secret for Anthropic API | `anthropic-api-key` |
| `S3_BUCKET` | S3 bucket for storing artifacts | Auto-generated |
| `TASK_TABLE_NAME` | DynamoDB table for task tracking | Auto-generated |
| `EVENT_BUS_NAME` | EventBridge bus name | `default` |
| `ENVIRONMENT` | Environment name | `production` |
| `MAX_ITERATIONS` | Max ReAct agent iterations | `10` |

### Lambda Configuration

- **Memory**: 1GB (1024 MB)
- **Timeout**: 10 minutes (600 seconds)
- **Runtime**: Python 3.11
- **Trigger**: EventBridge (on-demand)

## Event Structure

### Input Event (from Manager Agent)

```json
{
  "id": "event-123",
  "source": "manager.agent",
  "detail-type": "SystemIssue.high_error_rate",
  "detail": {
    "issue": {
      "type": "high_error_rate",
      "severity": "high",
      "description": "Error rate exceeded threshold",
      "affected_service": "API Gateway",
      "metrics": {
        "error_rate": 0.15
      }
    },
    "llm_analysis": {
      "severity": "high",
      "should_alert": true,
      "recommended_actions": [
        "Review recent deployments",
        "Check API Gateway configuration",
        "Implement retry logic"
      ],
      "root_cause_analysis": "Likely caused by timeout issues",
      "auto_remediation": "increase_timeout"
    },
    "context": {
      "environment": "production",
      "region": "us-east-1"
    }
  }
}
```

### Output Event (to QA Agent)

```json
{
  "source": "engineer.agent",
  "detail-type": "Implementation.bug_fix.Completed",
  "detail": {
    "task_id": "task-20240115-123456",
    "task_type": "bug_fix",
    "status": "success",
    "implementation_summary": "Fixed timeout issue by increasing Lambda timeout and adding retry logic",
    "pr_url": "https://github.com/org/repo/pull/123",
    "files_changed": [
      {
        "path": "src/api/handler.py",
        "action": "modified"
      }
    ],
    "tests_passed": true,
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

## Task Types

The Engineer Agent handles various implementation types:

- **BUG_FIX**: Fixes identified bugs or errors
- **INFRASTRUCTURE**: Updates Terraform configurations
- **CONFIGURATION**: Modifies application configurations
- **FEATURE**: Implements new features
- **REFACTORING**: Improves code structure
- **DOCUMENTATION**: Updates documentation

## Tools Available to ReAct Agent

1. **CodeAnalysisTool**: Analyzes existing codebase structure
2. **CodeGenerationTool**: Generates code using Claude
3. **TerraformTool**: Manages infrastructure configurations
4. **GitHubTool**: Performs GitHub operations (branches, commits, PRs)
5. **TestingTool**: Runs tests to validate implementations

## Implementation Flow

1. **Event Reception**: Receives event from EventBridge
2. **Task Analysis**: Determines task type and requirements
3. **Planning**: Creates detailed implementation plan
4. **Execution**: ReAct agent executes plan step-by-step
5. **Code Generation**: Claude generates necessary code
6. **GitHub Operations**: Creates branch and commits changes
7. **Pull Request**: Opens PR with implementation details
8. **Terraform Updates**: Updates infrastructure if needed
9. **Event Publication**: Notifies QA Agent for testing

## Monitoring

### CloudWatch Metrics

Monitor the following metrics in CloudWatch:

- **Invocations**: Number of times the function is triggered
- **Duration**: Execution time per invocation
- **Errors**: Number of failed executions
- **Throttles**: Number of throttled invocations

### CloudWatch Alarms

Pre-configured alarms:

- **Error Rate**: Triggers if errors > 10 in 5 minutes
- **Duration**: Triggers if average duration > 5 minutes
- **Throttles**: Triggers if throttles > 5 in 5 minutes

### Logs

View logs in CloudWatch Logs:

```bash
aws logs tail /aws/lambda/engineer-agent-production --follow
```

## Cost Estimation

Estimated monthly costs (based on typical usage):

| Component | Usage | Cost |
|-----------|-------|------|
| Lambda Invocations | ~100/month | $2.00 |
| Lambda Duration | ~5000 GB-seconds | $3.00 |
| DynamoDB | On-demand, light usage | $1.00 |
| S3 Storage | < 1GB | $0.10 |
| EventBridge | < 1M events | $1.00 |
| CloudWatch Logs | < 5GB | $2.50 |
| **Total** | | **~$10/month** |

## Troubleshooting

### Common Issues

1. **Lambda Timeout**
   - Increase timeout in Terraform configuration
   - Optimize agent iterations with `MAX_ITERATIONS`

2. **GitHub API Rate Limits**
   - Use GitHub App instead of personal token
   - Implement rate limit handling

3. **Large Dependencies**
   - Use container images for Lambda if layer > 250MB
   - Optimize requirements.txt

4. **Missing Permissions**
   - Check IAM role has all required permissions
   - Verify secrets are accessible

### Debug Mode

Enable verbose logging by setting environment variable:

```bash
export LANGCHAIN_VERBOSE=true
```

## Security Considerations

1. **Secrets Management**: All sensitive data stored in AWS Secrets Manager
2. **IAM Roles**: Least privilege principle applied
3. **GitHub Permissions**: Use fine-grained personal access tokens
4. **Code Review**: All changes go through PR process
5. **Audit Trail**: Complete task history in DynamoDB

## Development

### Local Testing

Test the Lambda function locally:

```python
# test_local.py
import json
from lambda_function import lambda_handler

event = {
    "source": "manager.agent",
    "detail-type": "SystemIssue.high_error_rate",
    "detail": {
        "issue": {
            "type": "high_error_rate",
            "description": "Test issue"
        }
    }
}

context = {}
response = lambda_handler(event, context)
print(json.dumps(response, indent=2))
```

### Adding New Tools

To add a new tool to the ReAct agent:

1. Create a new class extending `BaseTool`
2. Implement `_run` and `_arun` methods
3. Add to tools list in `_initialize_agent`

### Updating Dependencies

1. Modify `requirements.txt`
2. Run `deploy.sh` to rebuild packages
3. Update Lambda layer in AWS

## Integration with Other Agents

### Manager Agent Integration

The Engineer Agent receives events from Manager Agent when issues are detected or opportunities identified.

### QA Agent Integration

After implementation, the Engineer Agent publishes completion events that trigger the QA Agent to begin testing.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

[Your License Here]

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review CloudWatch logs
3. Contact the development team

## Roadmap

- [ ] Support for multiple programming languages
- [ ] Integration with more AI models
- [ ] Advanced code analysis capabilities
- [ ] Automated rollback on failures
- [ ] Performance optimization
- [ ] Multi-region support
