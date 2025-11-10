# Contributing to Omnilingual ASR

First off, thank you for considering contributing to Omnilingual ASR! We want to make contributing to this project as easy and transparent as possible. This project is a community effort, and we're excited to have you on board.

This document provides guidelines for contributing to the project. Please read it carefully to ensure a smooth and effective collaboration process.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
  - [Reporting Bugs](#reporting-bugs)
  - [Suggesting Enhancements](#suggesting-enhancements)
- [Contributor License Agreement (CLA)](#contributor-license-agreement-cla)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Style Guides](#style-guides)
  - [Coding Style](#coding-style)
  - [Git Commit Messages](#git-commit-messages)
- [License](#license)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](./CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How Can I Contribute?

### Reporting Bugs

We use GitHub Issues to track public bugs. Before filing a new issue, please search the existing [Issues](https://github.com/facebookresearch/omnilingual-asr/issues) to see if your problem has already been reported.

If you find a new bug, please file a new issue with a clear title and description, including:
- Steps to reproduce the issue.
- A minimal code sample that triggers the bug.
- Any relevant error messages or tracebacks.
- Your operating system and package versions.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe disclosure of security bugs. Please go through the process outlined on that page and **do not file a public issue** for security vulnerabilities.

### Suggesting Enhancements

If you have an idea for a new feature or an improvement to an existing one, we'd love to hear about it! Please open an issue to start a discussion.

## Contributor License Agreement (CLA)

In order to accept your pull request, we need you to submit a Contributor License Agreement (CLA). You only need to do this once to work on any of Meta's open-source projects.

**[Complete your CLA here](https://code.facebook.com/cla)**

## Development Setup

Omnilingual ASR is built on top of `fairseq2` and follows similar [guidelines](https://github.com/facebookresearch/fairseq2/blob/main/CONTRIBUTING.md).

1.  **Fork and clone the repository:**
    ```bash
    git clone https://github.com/your-username/omnilingual-asr.git
    cd omnilingual-asr
    ```

2.  **Set up a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3.  **Install the package in editable mode** with development and data dependencies:
    ```bash
    pip install -e ".[dev,data]"
    ```

4.  **Verify the installation** by running a quick import and the test suite:
    ```bash
    python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('Success!')"
    pytest
    ```

## Pull Request Process

We actively welcome your pull requests.

1.  **Create a branch:** Fork the repo and create your branch from `main`:
    ```bash
    git checkout -b feature/my-awesome-feature
    ```

2.  **Add code and tests:** Make your changes. If you've added code that should be tested, please add tests under `tests/unit/` or `tests/integration/`. Some tests can be slow (e.g., transcribing audio with large models) and may be marked with `@pytest.mark.slow` to avoid long CI runs.

3.  **Update documentation:** If you've changed APIs, please update the main `README.md` or any feature-specific READMEs.

4.  **Format and lint your code:** Ensure your code adheres to the project's style guidelines.
    ```bash
    isort .
    black .
    mypy .
    flake8 .
    ```

5.  **Run the test suite:** Ensure all tests pass.
    ```bash
    pytest tests/
    ```

6.  **Commit your changes:** Write a clear commit message describing what changed and why (see [Git Commit Messages](#git-commit-messages)).

7.  **Push and create a Pull Request:** Push your branch to your fork and submit a pull request to the `main` branch of the original repository.

8.  **Sign the CLA:** Ensure you have completed the [Contributor License Agreement](#contributor-license-agreement-cla). The automated CLA bot will check this on your PR.

## Style Guides

### Coding Style
*   Follow PEP 8 Python style guidelines.
*   Use 4 spaces for indentation (not tabs).
*   Maximum line length: 88 characters (enforced by Black).
*   Use type hints for all function signatures.
*   Write clear docstrings for public functions and classes.

### Git Commit Messages
We recommend following the [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) specification. This makes the commit history more readable.

A typical commit message looks like this:
```
feat: Add support for word-level timestamps in CTC models
```
Common prefixes include `feat`, `fix`, `docs`, `style`, `refactor`, `test`, and `chore`.

## License
By contributing to Omnilingual ASR, you agree that your contributions will be licensed under the [LICENSE](./LICENSE) file in the root directory of this source tree.
