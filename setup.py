from setuptools import setup, find_packages

setup(
    name="responsive-fine-tuner",
    version="0.1.0",
    description="Interactive tool for fine-tuning language models with human feedback",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "accelerate>=0.20.0",
        "peft>=0.4.0",
        "trl>=0.4.7",
        "gradio>=4.0.0",
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "python-dotenv>=0.19.0",
        "PyYAML>=6.0",
        "plotly>=5.0.0",
    ],
    author="Dyuti Dasmahapatra",
    author_email="dyutidasmahaptra@gmail.com",
)