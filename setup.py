from setuptools import setup, find_packages
import os
import sys

# Read version from bitsandbyes/__init__.py
with open(os.path.join("bitsandbyes", "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"\'')
            break
    else:
        version = "0.1.0"

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

# Define requirements
requirements = [
    "torch>=2.2.0",
    "numpy>=1.20.0",
]

setup(
    name="bits-and-byes",
    version=version,
    author="Bits-and-Byes Contributors",
    author_email="your-email@example.com",
    description="A library for efficient quantization and memory optimization in PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/bits-and-byes",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black",
            "isort",
            "flake8",
            "mypy",
        ],
        "docs": [
            "sphinx",
            "sphinx-rtd-theme",
        ],
    },
) 
