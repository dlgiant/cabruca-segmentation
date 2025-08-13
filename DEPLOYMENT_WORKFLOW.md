# Deployment Workflow Guide

## Overview

This repository uses a GitOps approach with environment-based deployments:
- **Main branch** → Production environment
- **All other branches** → Staging environment

## Branch Protection

The `main` branch is protected with the following rules:
- ❌ **No direct pushes allowed**
- ✅ **Pull requests required**
- ✅ **1 approval required**
- ✅ **Status checks must pass**
- ✅ **Conversations must be resolved**

## Environments

### Production Environment
- **Branch**: `main`
- **Config**: `configs/production.env`
- **Resources**: Higher CPU/Memory, multiple instances
- **Features**: Caching enabled, debug disabled
- **Deployment**: Automatic on merge to main

### Staging Environment
- **Branches**: `develop`, `feature/*`, `hotfix/*`, `release/*`
- **Config**: `configs/staging.env`
- **Resources**: Lower CPU/Memory, single instance
- **Features**: Debug enabled, profiling enabled
- **Deployment**: Automatic on push or PR

## Workflow Steps

### 1. Create a Feature Branch

```bash
# Start from develop branch
git checkout develop
git pull origin develop

# Create your feature branch
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Make your code changes
# Test locally
./deploy.sh  # This will deploy to staging

# Commit your changes
git add .
git commit -m "feat: add new feature"
```

### 3. Push to GitHub

```bash
git push origin feature/your-feature-name
```

This triggers:
- 🔧 CI pipeline (lint, test, security scan)
- 🚀 Automatic staging deployment
- 💬 PR comment with preview URLs

### 4. Create Pull Request

1. Go to GitHub repository
2. Click "New pull request"
3. Base: `develop` ← Compare: `feature/your-feature-name`
4. Fill out the PR template
5. Request review

### 5. Review Process

Reviewers will:
- Review code changes
- Test on staging environment
- Approve or request changes

### 6. Merge to Develop

After approval:
```bash
# Merge via GitHub UI or CLI
gh pr merge --squash
```

### 7. Deploy to Production

When ready for production:

```bash
# Create PR from develop to main
git checkout develop
git pull origin develop
gh pr create --base main --title "Release v1.x.x"
```

After approval and merge:
- 🚀 Automatic production deployment
- 📊 Production monitoring enabled

## Manual Deployment

If needed, you can deploy manually:

```bash
# Deploy based on current branch
./deploy.sh

# The script will automatically detect:
# - main branch → Production
# - other branches → Staging
```

## Setting Up Branch Protection

First-time setup only:

```bash
# Run the setup script
chmod +x setup-branch-protection.sh
./setup-branch-protection.sh
```

This will:
- Enable branch protection on `main`
- Create CODEOWNERS file
- Create PR template
- Document branching strategy

## GitHub Actions Workflows

### CI Pipeline (`ci.yml`)
Runs on: Pull requests and feature branches
- Linting (Black, isort, Flake8)
- Unit tests
- Security scanning
- Docker build test

### Staging Deployment (`staging-deploy.yml`)
Runs on: PRs and non-main branches
- Builds Docker image
- Pushes to ECR
- Updates ECS services
- Posts preview URLs to PR

### Production Deployment (`production-deploy.yml`)
Runs on: Push to main branch
- Builds production Docker image
- Pushes to ECR with production tags
- Updates production ECS services
- Runs health checks
- Sends deployment notifications

## Configuration Files

### Environment Variables

**Production** (`configs/production.env`):
- Higher resources (1024 CPU, 2048 Memory)
- Multiple instances (min: 2, max: 10)
- Caching enabled
- Debug disabled

**Staging** (`configs/staging.env`):
- Lower resources (256 CPU, 512 Memory)
- Single instance (min: 1, max: 3)
- Debug enabled
- Profiling enabled

## Rollback Procedure

If issues occur in production:

### Quick Rollback
```bash
# Revert the last deployment
aws ecs update-service \
  --cluster cabruca-prod-cluster \
  --service cabruca-prod-api \
  --task-definition <previous-task-definition> \
  --force-new-deployment
```

### Git Revert
```bash
# Create a revert commit
git checkout main
git pull origin main
git revert HEAD
git push origin main
```

## Monitoring

### Production Metrics
- CloudWatch Dashboard: Check AWS Console
- Logs: `/ecs/cabruca-prod/api`
- Alarms: Sent to `production-alerts@cabruca.com`

### Staging Metrics
- CloudWatch Dashboard: Check AWS Console
- Logs: `/ecs/cabruca-stg/api`
- Alarms: Sent to `staging-alerts@cabruca.com`

## Troubleshooting

### Deployment Failed
1. Check GitHub Actions logs
2. Review ECS service events
3. Check CloudWatch logs
4. Verify IAM permissions

### Service Unhealthy
1. Check target group health
2. Review container logs
3. Test endpoints manually
4. Check security groups

### Can't Push to Main
- Main branch is protected
- Create a PR instead
- Ensure CI checks pass
- Get required approvals

## Best Practices

1. **Always test on staging first**
2. **Keep PRs small and focused**
3. **Write descriptive commit messages**
4. **Update documentation**
5. **Monitor after deployment**
6. **Use feature flags for risky changes**

## Support

For issues or questions:
- Check CloudWatch logs
- Review GitHub Actions output
- Contact DevOps team
- Create an issue in GitHub