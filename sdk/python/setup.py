"""Setup for Prime-Sparse Python SDK."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="prime-sparse",
    version="1.0.0",
    author="Bradley Wallace",
    author_email="coo@koba42.com",
    description="Python SDK for Prime-Sparse neural network optimization API",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/koba42/prime-sparse",
    project_urls={
        "Documentation": "https://prime-sparse.readthedocs.io",
        "API Reference": "https://api.prime-sparse.com/docs",
        "Bug Tracker": "https://github.com/koba42/prime-sparse/issues",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "httpx>=0.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-asyncio>=0.21",
            "black",
            "mypy",
        ],
    },
    keywords=[
        "neural-network",
        "optimization",
        "sparsity",
        "machine-learning",
        "deep-learning",
        "model-compression",
    ],
)
