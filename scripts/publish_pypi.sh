#!/usr/bin/env bash
set -euo pipefail

# Build and publish to PyPI using twine. Requires TWINE_USERNAME and TWINE_PASSWORD
# or configure ~/.pypirc for secure upload.

python -m pip install --upgrade build twine
python -m build

echo "Built distributions in dist/"
echo "Uploading to PyPI (test by setting TWINE_REPOSITORY_URL or using TestPyPI)"

# To upload to TestPyPI:
# TWINE_USERNAME=<username> TWINE_PASSWORD=<password> python -m twine upload --repository testpypi dist/*

# To upload to production PyPI:
# TWINE_USERNAME=<username> TWINE_PASSWORD=<password> python -m twine upload dist/*

echo "Done."
