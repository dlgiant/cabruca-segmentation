# 🚀 GitHub Actions Setup Guide

## 🔴 Current Issue
The GitHub Actions workflows are failing because the required secrets are not configured in the repository.

## ✅ Quick Fix - Configure Required Secrets

### Option 1: Using the Setup Script (Recommended)
```bash
# Run the automated setup script
./setup-github-secrets.sh
```

This script will prompt you for:
- AWS Access Key ID
- AWS Secret Access Key
- AWS Account ID
- Alert Email
- AgentOps API Key (optional)
- Slack Webhook URL (optional)

### Option 2: Manual Configuration via GitHub UI

1. Go to your repository settings: https://github.com/dlgiant/cabruca-segmentation/settings/secrets/actions

2. Add the following repository secrets:

| Secret Name | Description | Required |
|------------|-------------|----------|
| `AWS_ACCESS_KEY_ID` | Your AWS access key | ✅ Yes |
| `AWS_SECRET_ACCESS_KEY` | Your AWS secret key | ✅ Yes |
| `AWS_ACCOUNT_ID` | Your AWS account ID (12 digits) | ✅ Yes |
| `ALERT_EMAIL` | Email for CloudWatch alerts | ✅ Yes |
| `AGENTOPS_API_KEY` | AgentOps API key for monitoring | ⚠️ Optional |
| `SLACK_WEBHOOK` | Slack webhook for notifications | ⚠️ Optional |

### Option 3: Using GitHub CLI
```bash
# Set AWS credentials
gh secret set AWS_ACCESS_KEY_ID
gh secret set AWS_SECRET_ACCESS_KEY
gh secret set AWS_ACCOUNT_ID
gh secret set ALERT_EMAIL

# Optional: Set AgentOps and Slack
gh secret set AGENTOPS_API_KEY
gh secret set SLACK_WEBHOOK
```

## 🔧 Fixed Issues in Workflows

### 1. Lambda Deploy Workflow
- **Issue**: JSON formatting error in change detection
- **Fix**: Added `-c` flag to `jq` for compact JSON output
- **File**: `.github/workflows/lambda-deploy.yml`

### 2. Main Deploy Workflow
- **Issue**: Missing permissions for GitHub deployments
- **Fix**: Added proper permissions block
- **File**: `.github/workflows/main-deploy.yml`

## 📝 Testing the Fixes

After setting up the secrets, you can test the workflows:

### 1. Test Lambda Deployment
```bash
# Deploy all Lambda functions
gh workflow run lambda-deploy.yml \
  -f environment=mvp \
  -f agents=all

# Check the status
gh run list --workflow=lambda-deploy.yml
```

### 2. Test Main Pipeline
```bash
# Run the main CI/CD pipeline
gh workflow run main-deploy.yml \
  -f environment=staging

# Check the status
gh run list --workflow=main-deploy.yml
```

### 3. Test Infrastructure Deployment
```bash
# First, plan the infrastructure
gh workflow run terraform-deploy.yml \
  -f action=plan \
  -f environment=mvp

# If plan looks good, apply
gh workflow run terraform-deploy.yml \
  -f action=apply \
  -f environment=mvp \
  -f auto_approve=true
```

## 🔍 Monitoring Workflow Runs

### View Recent Runs
```bash
# List all recent runs
gh run list

# List failed runs only
gh run list --status failure

# Watch a specific run
gh run watch
```

### View Logs
```bash
# View logs for a specific run
gh run view <run-id> --log

# View only failed job logs
gh run view <run-id> --log-failed
```

## 🚨 Troubleshooting

### If workflows still fail after setting secrets:

1. **Verify secrets are set correctly:**
```bash
gh secret list
```

2. **Check AWS credentials are valid:**
```bash
aws sts get-caller-identity
```

3. **Ensure AWS user has required permissions:**
- ECR access for Docker images
- Lambda deployment permissions
- DynamoDB access
- S3 bucket access
- CloudWatch logs access

4. **Check GitHub token permissions:**
- Ensure the workflow has `write` permissions for deployments
- Check branch protection rules aren't blocking

## 📊 Required AWS IAM Permissions

Your AWS IAM user/role needs these permissions:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "lambda:*",
        "dynamodb:*",
        "s3:*",
        "ecr:*",
        "logs:*",
        "events:*",
        "iam:PassRole",
        "secretsmanager:*"
      ],
      "Resource": "*"
    }
  ]
}
```

## ✅ Next Steps

1. Configure the GitHub secrets (use Option 1 above)
2. Re-run the failed workflows
3. Monitor the deployment progress
4. Once successful, your infrastructure will be deployed to AWS!

## 📧 Support

If you encounter issues:
1. Check the workflow logs: `gh run view --log`
2. Review this guide
3. Check AWS CloudTrail for permission issues
4. Open an issue in the repository