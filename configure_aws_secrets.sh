#!/bin/bash

# AWS Secrets Configuration Script for GitHub Actions
set -e

echo "========================================="
echo "🔐 Configuring AWS Secrets for GitHub Actions"
echo "========================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated with GitHub
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub"
    echo "Please run: gh auth login"
    exit 1
fi

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI is not installed"
    echo "Please install it from: https://aws.amazon.com/cli/"
    exit 1
fi

echo ""
echo "📝 Getting AWS credentials from your local configuration..."

# Get AWS credentials
AWS_ACCESS_KEY_ID=$(aws configure get aws_access_key_id)
AWS_SECRET_ACCESS_KEY=$(aws configure get aws_secret_access_key)
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region || echo "sa-east-1")

if [ -z "$AWS_ACCESS_KEY_ID" ] || [ -z "$AWS_SECRET_ACCESS_KEY" ]; then
    echo "❌ AWS credentials not found in local configuration"
    echo "Please configure AWS CLI first: aws configure"
    exit 1
fi

echo "✅ Found AWS credentials for account: $AWS_ACCOUNT_ID"
echo "📍 Region: $AWS_REGION"

# Get email for alerts
echo ""
read -p "Enter email address for CloudWatch alerts: " ALERT_EMAIL
if [ -z "$ALERT_EMAIL" ]; then
    ALERT_EMAIL="alerts@example.com"
    echo "Using default: $ALERT_EMAIL"
fi

# Optional: AgentOps API Key
echo ""
echo "📊 AgentOps Configuration (Optional - press Enter to skip)"
read -p "Enter AgentOps API Key: " AGENTOPS_API_KEY

# Optional: Slack Webhook
echo ""
echo "💬 Slack Configuration (Optional - press Enter to skip)"
read -p "Enter Slack Webhook URL: " SLACK_WEBHOOK

echo ""
echo "========================================="
echo "🚀 Setting GitHub Secrets..."
echo "========================================="

# Set required secrets
echo "Setting AWS_ACCESS_KEY_ID..."
echo "$AWS_ACCESS_KEY_ID" | gh secret set AWS_ACCESS_KEY_ID

echo "Setting AWS_SECRET_ACCESS_KEY..."
echo "$AWS_SECRET_ACCESS_KEY" | gh secret set AWS_SECRET_ACCESS_KEY

echo "Setting AWS_ACCOUNT_ID..."
echo "$AWS_ACCOUNT_ID" | gh secret set AWS_ACCOUNT_ID

echo "Setting ALERT_EMAIL..."
echo "$ALERT_EMAIL" | gh secret set ALERT_EMAIL

# Set optional secrets if provided
if [ ! -z "$AGENTOPS_API_KEY" ]; then
    echo "Setting AGENTOPS_API_KEY..."
    echo "$AGENTOPS_API_KEY" | gh secret set AGENTOPS_API_KEY
fi

if [ ! -z "$SLACK_WEBHOOK" ]; then
    echo "Setting SLACK_WEBHOOK..."
    echo "$SLACK_WEBHOOK" | gh secret set SLACK_WEBHOOK
fi

# Set GitHub token (uses current gh auth token)
echo "Setting GH_TOKEN from current authentication..."
GH_TOKEN=$(gh auth token)
echo "$GH_TOKEN" | gh secret set GH_TOKEN

echo ""
echo "========================================="
echo "✅ GitHub Secrets Configuration Complete!"
echo "========================================="

echo ""
echo "📋 Configured secrets:"
gh secret list

echo ""
echo "🌍 Creating GitHub Environments..."

# Create environments
for ENV in mvp staging production; do
    echo "Creating environment: $ENV"
    
    # Create environment using GitHub API
    gh api \
        --method PUT \
        -H "Accept: application/vnd.github+json" \
        "/repos/$(gh repo view --json nameWithOwner -q .nameWithOwner)/environments/$ENV" \
        -f wait_timer=0 \
        -f deployment_branch_policy='{"protected_branches":false,"custom_branch_policies":true}' \
        2>/dev/null || echo "Environment $ENV exists or created"
done

echo ""
echo "========================================="
echo "🎉 Setup Complete!"
echo "========================================="

echo ""
echo "📝 Next steps:"
echo "1. Re-run any failed workflows:"
echo "   gh workflow run main-deploy.yml -f environment=mvp"
echo ""
echo "2. Deploy infrastructure:"
echo "   gh workflow run terraform-deploy.yml -f action=plan -f environment=mvp"
echo ""
echo "3. Deploy Lambda functions:"
echo "   gh workflow run lambda-deploy.yml -f environment=mvp -f agents=all"
echo ""
echo "4. Monitor deployment:"
echo "   gh run list"
echo "   gh run watch"

echo ""
echo "🔍 To verify secrets are working:"
echo "   gh workflow run main-deploy.yml -f environment=mvp"
echo "   gh run view --log"