# Branching Strategy

## Branch Protection
- **main**: Protected branch, requires PR and approval
- Direct pushes to main are **blocked**
- All changes must go through pull requests

## Branch Naming Convention

### Main Branches
- `main` - Production-ready code (uses production settings)
- `develop` - Integration branch for features (uses staging settings)

### Supporting Branches
- `feature/*` - New features (uses staging settings)
  - Example: `feature/user-authentication`
  - Example: `feature/add-segmentation-model`

- `hotfix/*` - Emergency fixes for production
  - Example: `hotfix/fix-api-timeout`
  - Example: `hotfix/security-patch`

- `release/*` - Release preparation
  - Example: `release/v1.2.0`

- `bugfix/*` - Bug fixes for develop branch
  - Example: `bugfix/fix-memory-leak`

## Workflow

1. Create a new branch from `develop`:
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

3. Push to GitHub:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a Pull Request to `develop`
   - Automated staging deployment will run
   - Tests must pass
   - Code review required

5. After approval, merge to `develop`

6. When ready for production:
   - Create PR from `develop` to `main`
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

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Test additions or changes
- `chore:` Build process or auxiliary tool changes
- `perf:` Performance improvements

Example: `feat: add image segmentation endpoint`
