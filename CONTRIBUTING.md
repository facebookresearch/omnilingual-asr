# Contributing to Omnilingual ASR
We want to make contributing to this project as easy and transparent as
possible.

## Our Development Process
Omnilingual ASR is built on top of fairseq2 and follows similar [guidelines](https://github.com/facebookresearch/fairseq2/blob/main/CONTRIBUTING.md).

1. Fork and clone the repository:
```bash
git clone https://github.com/facebookresearch/omnilingual-asr.git
cd omnilingual-asr
```

2. Install the package in development mode:
```bash
pip install -e ".[dev,data]"
```

3. Verify the installation:
```bash
python -c "from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline; print('Success!')"
pytest
```


## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`:
```bash
git checkout -b feature_name
```
2. If you've added code that should be tested, add tests. Consider both unit and integration tests (under `tests/unit/` and `tests/integration/` folders). Some tests can be slow (e.g., transcribing audio with large models) and may be marked with `@pytest.mark.slow` to avoid long CI runs.
3. If you've changed APIs, update the documentation in the main README.md or feature-specific READMEs.
4. Ensure the test suite passes.
```bash
pytest tests/
```
5. **Format** your code and make sure it passes linting:
```bash
isort . && black .
mypy && flake8 .
```
6. **Commit**: Write clear commit messages describing what changed and why

7. **Push and Create PR**: Push to your fork and submit a pull request to `main`

8. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* Follow PEP 8 Python style guidelines
* Use 4 spaces for indentation (not tabs)
* Maximum line length: 88 characters (Black formatter default)
* Use type hints for function signatures
* Write docstrings for public functions and classes

## License
By contributing to Omnilingual ASR, you agree that your contributions will be licensed under the LICENSE file in the root directory of this source tree.
