#!/bin/bash

# Setup Branch Protection Rules for Main Branch
# This script configures branch protection using GitHub CLI

set -e

echo "🔒 Setting up branch protection rules for main branch..."

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed. Please install it first:"
    echo "   brew install gh"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub. Please run:"
    echo "   gh auth login"
    exit 1
fi

# Get repository information
REPO_OWNER=$(gh repo view --json owner -q .owner.login)
REPO_NAME=$(gh repo view --json name -q .name)

echo "Repository: $REPO_OWNER/$REPO_NAME"

# Configure branch protection for main branch
echo "Configuring protection rules for 'main' branch..."

# Enable branch protection via GitHub API
# For personal repositories, we use a simpler configuration
gh api \
  --method PUT \
  -H "Accept: application/vnd.github+json" \
  "/repos/$REPO_OWNER/$REPO_NAME/branches/main/protection" \
  --input - <<EOF
{
  "required_status_checks": {
    "strict": true,
    "contexts": []
  },
  "enforce_admins": false,
  "required_pull_request_reviews": {
    "dismiss_stale_reviews": true,
    "require_code_owner_reviews": false,
    "required_approving_review_count": 1,
    "require_last_push_approval": false
  },
  "restrictions": null,
  "allow_force_pushes": false,
  "allow_deletions": false,
  "required_conversation_resolution": true
}
EOF

echo "✅ Branch protection enabled for 'main' branch"

# Create CODEOWNERS file
echo "Creating CODEOWNERS file..."
cat > .github/CODEOWNERS <<EOF
# Code Owners for Cabruca Segmentation Project
# These owners will be requested for review when someone opens a pull request

# Global owners
* @ricardonunes

# Infrastructure and deployment
/terraform/ @ricardonunes
/.github/ @ricardonunes
/configs/ @ricardonunes

# Application code
/src/ @ricardonunes
/api_server.py @ricardonunes
/streamlit_app.py @ricardonunes

# Documentation
*.md @ricardonunes
/docs/ @ricardonunes
EOF

echo "✅ CODEOWNERS file created"

# Create pull request template
echo "Creating pull request template..."
mkdir -p .github
cat > .github/pull_request_template.md <<EOF
## Description
Brief description of the changes in this PR

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Unit tests pass locally
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Staging deployment successful

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published

## Screenshots (if applicable)
Add screenshots to help explain your changes

## Additional Notes
Any additional information that reviewers should know
EOF

echo "✅ Pull request template created"

# Create branch naming convention document
cat > BRANCHING_STRATEGY.md <<EOF
# Branching Strategy

## Branch Protection
- **main**: Protected branch, requires PR and approval
- Direct pushes to main are **blocked**
- All changes must go through pull requests

## Branch Naming Convention

### Main Branches
- \`main\` - Production-ready code (uses production settings)
- \`develop\` - Integration branch for features (uses staging settings)

### Supporting Branches
- \`feature/*\` - New features (uses staging settings)
  - Example: \`feature/user-authentication\`
  - Example: \`feature/add-segmentation-model\`

- \`hotfix/*\` - Emergency fixes for production
  - Example: \`hotfix/fix-api-timeout\`
  - Example: \`hotfix/security-patch\`

- \`release/*\` - Release preparation
  - Example: \`release/v1.2.0\`

- \`bugfix/*\` - Bug fixes for develop branch
  - Example: \`bugfix/fix-memory-leak\`

## Workflow

1. Create a new branch from \`develop\`:
   \`\`\`bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   \`\`\`

2. Make your changes and commit:
   \`\`\`bash
   git add .
   git commit -m "feat: add new feature"
   \`\`\`

3. Push to GitHub:
   \`\`\`bash
   git push origin feature/your-feature-name
   \`\`\`

4. Create a Pull Request to \`develop\`
   - Automated staging deployment will run
   - Tests must pass
   - Code review required

5. After approval, merge to \`develop\`

6. When ready for production:
   - Create PR from \`develop\` to \`main\`
   - Production deployment runs automatically after merge

## Environment Mapping

| Branch Pattern | Environment | Configuration File |
|---------------|-------------|-------------------|
| main          | Production  | configs/production.env |
| develop       | Staging     | configs/staging.env |
| feature/*     | Staging     | configs/staging.env |
| hotfix/*      | Staging     | configs/staging.env |
| release/*     | Staging     | configs/staging.env |

## Commit Message Convention

Follow the Conventional Commits specification:

- \`feat:\` New feature
- \`fix:\` Bug fix
- \`docs:\` Documentation changes
- \`style:\` Code style changes (formatting, etc.)
- \`refactor:\` Code refactoring
- \`test:\` Test additions or changes
- \`chore:\` Build process or auxiliary tool changes
- \`perf:\` Performance improvements

Example: \`feat: add image segmentation endpoint\`
EOF

echo "✅ Branching strategy document created"

echo ""
echo "🎉 Branch protection setup complete!"
echo ""
echo "Summary of changes:"
echo "  ✅ Main branch protection enabled"
echo "  ✅ Requires pull request reviews (1 approval)"
echo "  ✅ Requires status checks to pass"
echo "  ✅ Dismisses stale reviews"
echo "  ✅ Requires conversation resolution"
echo "  ✅ CODEOWNERS file created"
echo "  ✅ Pull request template created"
echo "  ✅ Branching strategy documented"
echo ""
echo "Next steps:"
echo "  1. Commit these changes to a feature branch"
echo "  2. Create a PR to main to apply the configuration"
echo "  3. After merge, the protection rules will be active"