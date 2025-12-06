# Quick Start

This quick start shows how to install and run the project locally after packaging or from source.

From source (recommended for development):

```bash
git clone https://github.com/dyra-12/Responsive-Fine-Tuner.git
cd Responsive-Fine-Tuner
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=. python run_app.py --port 7860
```

Install from PyPI (once published):

```bash
pip install responsive-fine-tuner
rft --help
```

Run with Docker (build locally):

```bash
docker build -t responsive-fine-tuner:local -f deployment/Dockerfile .
docker run -p 7860:7860 responsive-fine-tuner:local
```

Notes
- Publishing to PyPI and Docker Hub requires credentials (see `scripts/`).
- The CLI `rft` will be available after installation and maps to the project's `run_app.py`.
