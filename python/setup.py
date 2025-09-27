#!/usr/bin/env python3
"""
Setup script for QAOA Portfolio Optimizer Python package
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, "..", "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Read requirements
with open("requirements.txt") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="qaoa-portfolio",
    version="0.1.0",
    author="dasobral",
    author_email="dasobral93@gmail.com",
    description="QAOA-based portfolio optimization with quantum-inspired algorithms",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/qaoa-portfolio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Office/Business :: Financial :: Investment",
        "License :: Other/Proprietary License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
        "jupyter": [
            "jupyter>=1.0.0",
            "ipykernel>=6.25.0",
            "ipywidgets>=8.0.0",
        ],
        "profiling": [
            "line-profiler>=4.1.0",
            "memory-profiler>=0.61.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qaoa-portfolio=qaoa_portfolio.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "qaoa_portfolio": ["data/*.csv", "config/*.json"],
    },
)
