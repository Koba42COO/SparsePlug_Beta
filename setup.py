from setuptools import setup, find_packages

setup(
    name="upg-pac",
    version="5.0.0",
    description="Unified Prime-Sparse Quantized Platform (V5)",
    author="UPG-PAC Team",
    author_email="opensource@koba42.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy",
        "fastapi",
        "uvicorn",
        "requests",
        "transformers" # Optional, but common
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)
