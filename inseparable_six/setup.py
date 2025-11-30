#!/usr/bin/env python3
"""
Setup script for inseparable_six package.

Installation:
    cd inseparable_six
    pip install -e .

Or from parent directory:
    pip install -e ./inseparable_six
"""

from setuptools import setup, find_packages

setup(
    name="inseparable_six",
    version="0.1.0",
    author="Inseparable Six Team",
    description="Layer-Adaptive Differential Privacy for LLM Fine-Tuning",
    long_description=open("README.md").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(where="."),
    package_dir={"": "."},
    py_modules=[
        "config",
        "metrics", 
        "allocation_strategies",
        "baseline",
        "layer_adaptive_optimizer",
        "run_experiments",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.2.0",
        "numpy>=1.15,<2",
        "scipy>=1.2",
        "opt-einsum>=3.3.0",
        "tqdm>=4.40",
        "datasets",
        "transformers",
        "peft",
        "opacus",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black",
            "isort",
        ],
    },
    entry_points={
        "console_scripts": [
            "run-dp-experiments=run_experiments:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
