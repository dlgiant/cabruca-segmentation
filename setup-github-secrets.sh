#!/bin/bash

# GitHub Secrets Setup Script
# This script helps set up required GitHub secrets for CI/CD pipelines

set -e

echo "========================================="
echo "🔐 GitHub Secrets Setup for CI/CD"
echo "========================================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed"
    echo "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub"
    echo "Please run: gh auth login"
    exit 1
fi

# Get repository information
REPO=$(gh repo view --json nameWithOwner -q .nameWithOwner)
echo "📦 Repository: $REPO"

# Function to set secret
set_secret() {
    local SECRET_NAME=$1
    local SECRET_VALUE=$2
    local ENVIRONMENT=$3
    
    if [ -z "$ENVIRONMENT" ]; then
        # Repository secret
        echo "Setting repository secret: $SECRET_NAME"
        echo "$SECRET_VALUE" | gh secret set "$SECRET_NAME" --repo "$REPO"
    else
        # Environment secret
        echo "Setting environment secret: $SECRET_NAME for $ENVIRONMENT"
        echo "$SECRET_VALUE" | gh secret set "$SECRET_NAME" --env "$ENVIRONMENT" --repo "$REPO"
    fi
}

echo ""
echo "📝 Setting up AWS credentials..."

# AWS Credentials
read -p "Enter AWS Access Key ID: " AWS_ACCESS_KEY_ID
read -s -p "Enter AWS Secret Access Key: " AWS_SECRET_ACCESS_KEY
echo ""
read -p "Enter AWS Account ID: " AWS_ACCOUNT_ID
read -p "Enter Alert Email: " ALERT_EMAIL

# Set repository-wide secrets
set_secret "AWS_ACCESS_KEY_ID" "$AWS_ACCESS_KEY_ID"
set_secret "AWS_SECRET_ACCESS_KEY" "$AWS_SECRET_ACCESS_KEY"
set_secret "AWS_ACCOUNT_ID" "$AWS_ACCOUNT_ID"
set_secret "ALERT_EMAIL" "$ALERT_EMAIL"

echo ""
echo "📝 Setting up AgentOps API key..."

# AgentOps API Key (retrieve from AWS Secrets Manager if available)
if command -v aws &> /dev/null; then
    echo "Attempting to retrieve AgentOps API key from AWS..."
    AGENTOPS_KEY=$(aws secretsmanager get-secret-value \
        --secret-id "cabruca-mvp-agentops-key" \
        --region sa-east-1 \
        --query 'SecretString' \
        --output text 2>/dev/null || echo "")
    
    if [ -z "$AGENTOPS_KEY" ]; then
        read -p "Enter AgentOps API Key: " AGENTOPS_KEY
    else
        echo "✅ Retrieved AgentOps key from AWS Secrets Manager"
    fi
else
    read -p "Enter AgentOps API Key: " AGENTOPS_KEY
fi

set_secret "AGENTOPS_API_KEY" "$AGENTOPS_KEY"

echo ""
echo "📝 Setting up optional integrations..."

# Slack Webhook (optional)
read -p "Enter Slack Webhook URL (press Enter to skip): " SLACK_WEBHOOK
if [ ! -z "$SLACK_WEBHOOK" ]; then
    set_secret "SLACK_WEBHOOK" "$SLACK_WEBHOOK"
fi

# GitHub Token (use existing from gh CLI)
echo "Using GitHub token from gh CLI authentication..."
GH_TOKEN=$(gh auth token)
set_secret "GH_TOKEN" "$GH_TOKEN"

echo ""
echo "🌍 Creating GitHub environments..."

# Create environments
for ENV in mvp staging production; do
    echo "Creating environment: $ENV"
    
    # Create environment using GitHub API
    gh api \
        --method PUT \
        -H "Accept: application/vnd.github+json" \
        "/repos/$REPO/environments/$ENV" \
        -f wait_timer=0 \
        -f deployment_branch_policy='{"protected_branches":false,"custom_branch_policies":true}' \
        2>/dev/null || echo "Environment $ENV already exists or created"
    
    # Set environment-specific secrets if needed
    if [ "$ENV" == "production" ]; then
        # Add production-specific protections
        gh api \
            --method PUT \
            -H "Accept: application/vnd.github+json" \
            "/repos/$REPO/environments/$ENV" \
            -f wait_timer=5 \
            -f reviewers='[]' \
            -f deployment_branch_policy='{"protected_branches":true,"custom_branch_policies":false}' \
            2>/dev/null || true
    fi
done

echo ""
echo "✅ GitHub secrets setup complete!"
echo ""
echo "📋 Summary of configured secrets:"
echo "  - AWS_ACCESS_KEY_ID"
echo "  - AWS_SECRET_ACCESS_KEY"
echo "  - AWS_ACCOUNT_ID"
echo "  - ALERT_EMAIL"
echo "  - AGENTOPS_API_KEY"
echo "  - GH_TOKEN"
if [ ! -z "$SLACK_WEBHOOK" ]; then
    echo "  - SLACK_WEBHOOK"
fi

echo ""
echo "🌍 Environments created:"
echo "  - mvp"
echo "  - staging"
echo "  - production"

echo ""
echo "📝 Next steps:"
echo "1. Review and merge the GitHub Actions workflows"
echo "2. Create a test PR to validate the PR checks"
echo "3. Deploy infrastructure using: gh workflow run terraform-deploy.yml"
echo "4. Deploy Lambda functions using: gh workflow run lambda-deploy.yml"

echo ""
echo "🚀 To trigger a deployment manually:"
echo "  gh workflow run main-deploy.yml -f environment=mvp"