# Contributing to Cabruca Segmentation

Thank you for your interest in contributing to the Cabruca Segmentation project\! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Describe the issue clearly with steps to reproduce
- Include system information (OS, Python version, GPU)
- Attach relevant logs or error messages

### Suggesting Features
- Open a discussion in GitHub Discussions
- Explain the use case and benefits
- Provide examples if possible

### Submitting Pull Requests
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest tests/`)
5. Commit with clear messages (`git commit -m 'Add amazing feature'`)
6. Push to your branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/cabruca-segmentation.git
cd cabruca-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Run tests
pytest tests/
```

## Code Style
- Follow PEP 8
- Use type hints where appropriate
- Document functions with docstrings
- Keep lines under 100 characters

## Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for >80% code coverage

## Documentation
- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Update API documentation if needed

## License
By contributing, you agree that your contributions will be licensed under the MIT License.
EOF < /dev/null