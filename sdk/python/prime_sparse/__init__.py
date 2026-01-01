"""
Prime-Sparse Python SDK
=======================

Easy-to-use SDK for the Prime-Sparse API.

Installation:
    pip install prime-sparse

Quick Start:
    from prime_sparse import Client

    client = Client(api_key="your_api_key")
    
    # Optimize a model
    result = client.optimize("model.pt", sparsity=0.96)
    print(f"Optimized to {result.size_mb}MB")
"""

__version__ = "1.0.0"
__author__ = "Bradley Wallace"

from .client import Client, AsyncClient
from .models import (
    Model,
    OptimizationJob,
    OptimizationConfig,
    OptimizationResult,
    UsageSummary,
)
from .exceptions import (
    PrimeSparseError,
    AuthenticationError,
    RateLimitError,
    ValidationError,
    OptimizationError,
)

__all__ = [
    "Client",
    "AsyncClient",
    "Model",
    "OptimizationJob",
    "OptimizationConfig",
    "OptimizationResult",
    "UsageSummary",
    "PrimeSparseError",
    "AuthenticationError",
    "RateLimitError",
    "ValidationError",
    "OptimizationError",
]
