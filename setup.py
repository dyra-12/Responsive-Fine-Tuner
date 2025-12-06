from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

def parse_requirements(path="requirements.txt"):
    try:
        with open(path, "r", encoding="utf-8") as fh:
            lines = [l.strip() for l in fh if l.strip() and not l.strip().startswith("#")]
        return lines
    except Exception:
        return []

setup(
    name="responsive-fine-tuner",
    version="0.1.0",
    description="Interactive tool for fine-tuning language models with human feedback",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dyra-12/Responsive-Fine-Tuner",
    author="dyra-12",
    license="MIT",
    packages=find_packages(exclude=("tests", "docs", "deployment")),
    include_package_data=True,
    install_requires=parse_requirements(),
    entry_points={
        'console_scripts': [
            'rft=rft.cli:main'
        ]
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)