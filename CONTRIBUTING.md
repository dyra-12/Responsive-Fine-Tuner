# Contributing to Responsive Fine-Tuner

Thanks for your interest in contributing! We welcome contributions of all kinds: bug reports, documentation fixes, tests, and new features.

Getting started
- Fork the repository and create a topic branch for your change.
- Keep changes focused and submit small PRs where possible.

Development environment
1. Create and activate a virtualenv:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Run tests locally:

```bash
PYTHONPATH=. pytest -q
```

Code style
- Use readable, well-documented code. Keep changes consistent with existing style.
- Use descriptive commit messages and fill PR descriptions with the rationale for changes.

Testing
- Add unit tests for any new behavior or bug fix.
- Prefer small, focused tests; use `pytest`.

Submitting a pull request
1. Open a PR against `main` with a clear title and description.
2. Include tests where appropriate and ensure CI checks pass.
3. Link relevant issues and provide reproduction steps if fixing a bug.

Security
- For security issues, please open a private issue or contact the maintainers directly instead of public channels.

Thanks for helping make this project better!
