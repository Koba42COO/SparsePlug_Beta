"""
Prime-Sparse API Client
=======================

Synchronous and async clients for the Prime-Sparse API.
"""

import os
import time
from pathlib import Path
from typing import Optional, Dict, Any, Union, BinaryIO
from dataclasses import dataclass

import httpx

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


DEFAULT_API_URL = "https://api.prime-sparse.com/api/v1"
DEFAULT_TIMEOUT = 300  # 5 minutes


class Client:
    """
    Synchronous client for Prime-Sparse API.
    
    Example:
        client = Client(api_key="ps_live_xxx")
        
        # Upload and optimize
        model = client.upload_model("model.pt")
        job = client.optimize(model.id, sparsity=0.96)
        
        # Wait for completion
        result = job.wait()
        result.download("optimized.safetensors")
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        """
        Initialize the client.
        
        Args:
            api_key: Your API key (or set PRIME_SPARSE_API_KEY env var)
            api_url: API base URL (default: https://api.prime-sparse.com/api/v1)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key or os.getenv("PRIME_SPARSE_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required. Set PRIME_SPARSE_API_KEY or pass api_key.")
        
        self.api_url = api_url or os.getenv("PRIME_SPARSE_API_URL", DEFAULT_API_URL)
        self.timeout = timeout
        
        self._client = httpx.Client(
            base_url=self.api_url,
            headers={"X-API-Key": self.api_key},
            timeout=timeout,
        )
    
    def _request(
        self,
        method: str,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Make API request."""
        try:
            response = self._client.request(method, path, **kwargs)
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", 60)
                raise RateLimitError(f"Rate limited. Retry after {retry_after}s")
            elif response.status_code == 422:
                raise ValidationError(response.json().get("detail", "Validation error"))
            elif response.status_code >= 400:
                raise PrimeSparseError(f"API error: {response.text}")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise PrimeSparseError(f"Request failed: {e}")
    
    # Model methods
    
    def upload_model(
        self,
        file_path: Union[str, Path],
        name: Optional[str] = None,
    ) -> Model:
        """
        Upload a model file.
        
        Args:
            file_path: Path to model file
            name: Optional name for the model
        
        Returns:
            Model object
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValidationError(f"File not found: {file_path}")
        
        # Get presigned upload URL
        filename = name or file_path.name
        size_bytes = file_path.stat().st_size
        
        response = self._request(
            "POST",
            "/models/upload-url",
            json={"filename": filename, "size_bytes": size_bytes},
        )
        
        upload_url = response["upload_url"]
        model_id = response["model_id"]
        
        # Upload file
        with open(file_path, "rb") as f:
            upload_response = httpx.put(
                upload_url,
                content=f,
                headers={"Content-Type": "application/octet-stream"},
                timeout=self.timeout,
            )
            upload_response.raise_for_status()
        
        # Confirm upload
        confirm_response = self._request(
            "POST",
            f"/models/{model_id}/confirm",
        )
        
        return Model(**confirm_response)
    
    def list_models(self) -> list:
        """List all user's models."""
        response = self._request("GET", "/models")
        return [Model(**m) for m in response.get("models", [])]
    
    def get_model(self, model_id: str) -> Model:
        """Get model details."""
        response = self._request("GET", f"/models/{model_id}")
        return Model(**response)
    
    def delete_model(self, model_id: str) -> bool:
        """Delete a model."""
        self._request("DELETE", f"/models/{model_id}")
        return True
    
    # Optimization methods
    
    def optimize(
        self,
        model_id: str,
        sparsity: float = 0.96,
        preserve_accuracy: bool = True,
        output_format: str = "safetensors",
        quantize: bool = False,
        wait: bool = False,
    ) -> OptimizationJob:
        """
        Create optimization job.
        
        Args:
            model_id: ID of model to optimize
            sparsity: Target sparsity (0.5-0.99)
            preserve_accuracy: Whether to validate accuracy
            output_format: Output format (safetensors, pytorch, onnx)
            quantize: Apply INT8 quantization
            wait: Wait for job to complete
        
        Returns:
            OptimizationJob object
        """
        config = OptimizationConfig(
            target_sparsity=sparsity,
            preserve_accuracy=preserve_accuracy,
            output_format=output_format,
            quantize=quantize,
        )
        
        response = self._request(
            "POST",
            "/optimize",
            json={
                "model_id": model_id,
                "config": config.__dict__,
            },
        )
        
        job = OptimizationJob(client=self, **response)
        
        if wait:
            job.wait()
        
        return job
    
    def get_job(self, job_id: str) -> OptimizationJob:
        """Get optimization job status."""
        response = self._request("GET", f"/optimize/{job_id}")
        return OptimizationJob(client=self, **response)
    
    def list_jobs(self) -> list:
        """List all optimization jobs."""
        response = self._request("GET", "/optimize")
        return [OptimizationJob(client=self, **j) for j in response.get("jobs", [])]
    
    # Usage methods
    
    def get_usage(self) -> UsageSummary:
        """Get current usage summary."""
        response = self._request("GET", "/billing/usage")
        return UsageSummary(**response)
    
    # Convenience methods
    
    def optimize_file(
        self,
        file_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        sparsity: float = 0.96,
        **kwargs
    ) -> OptimizationResult:
        """
        Convenience method: upload, optimize, and download in one call.
        
        Args:
            file_path: Path to model file
            output_path: Where to save optimized model
            sparsity: Target sparsity
            **kwargs: Additional optimization options
        
        Returns:
            OptimizationResult
        """
        # Upload
        model = self.upload_model(file_path)
        
        # Optimize
        job = self.optimize(model.id, sparsity=sparsity, **kwargs)
        
        # Wait for completion
        result = job.wait()
        
        # Download if path specified
        if output_path and result.download_url:
            result.download(output_path)
        
        return result
    
    def close(self):
        """Close the client."""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


class AsyncClient:
    """
    Async client for Prime-Sparse API.
    
    Example:
        async with AsyncClient(api_key="ps_live_xxx") as client:
            model = await client.upload_model("model.pt")
            job = await client.optimize(model.id, sparsity=0.96)
            result = await job.wait()
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
    ):
        self.api_key = api_key or os.getenv("PRIME_SPARSE_API_KEY")
        if not self.api_key:
            raise AuthenticationError("API key required")
        
        self.api_url = api_url or os.getenv("PRIME_SPARSE_API_URL", DEFAULT_API_URL)
        self.timeout = timeout
        
        self._client = httpx.AsyncClient(
            base_url=self.api_url,
            headers={"X-API-Key": self.api_key},
            timeout=timeout,
        )
    
    async def _request(self, method: str, path: str, **kwargs) -> Dict[str, Any]:
        """Make async API request."""
        try:
            response = await self._client.request(method, path, **kwargs)
            
            if response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 429:
                raise RateLimitError("Rate limited")
            elif response.status_code >= 400:
                raise PrimeSparseError(f"API error: {response.text}")
            
            return response.json()
            
        except httpx.RequestError as e:
            raise PrimeSparseError(f"Request failed: {e}")
    
    async def upload_model(self, file_path: Union[str, Path]) -> Model:
        """Upload model asynchronously."""
        file_path = Path(file_path)
        
        response = await self._request(
            "POST",
            "/models/upload-url",
            json={
                "filename": file_path.name,
                "size_bytes": file_path.stat().st_size,
            },
        )
        
        upload_url = response["upload_url"]
        model_id = response["model_id"]
        
        with open(file_path, "rb") as f:
            async with httpx.AsyncClient() as upload_client:
                await upload_client.put(
                    upload_url,
                    content=f.read(),
                    headers={"Content-Type": "application/octet-stream"},
                )
        
        confirm_response = await self._request("POST", f"/models/{model_id}/confirm")
        return Model(**confirm_response)
    
    async def optimize(self, model_id: str, sparsity: float = 0.96, **kwargs) -> OptimizationJob:
        """Create optimization job asynchronously."""
        response = await self._request(
            "POST",
            "/optimize",
            json={
                "model_id": model_id,
                "config": {"target_sparsity": sparsity, **kwargs},
            },
        )
        return OptimizationJob(client=self, **response)
    
    async def get_usage(self) -> UsageSummary:
        """Get usage summary asynchronously."""
        response = await self._request("GET", "/billing/usage")
        return UsageSummary(**response)
    
    async def close(self):
        """Close the client."""
        await self._client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        await self.close()
