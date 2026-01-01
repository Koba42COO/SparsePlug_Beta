"""
Prime-Sparse SDK Models
=======================

Data models for API responses.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Dict, Any, TYPE_CHECKING
from pathlib import Path

import httpx


if TYPE_CHECKING:
    from .client import Client


@dataclass
class OptimizationConfig:
    """Configuration for optimization job."""
    target_sparsity: float = 0.96
    preserve_accuracy: bool = True
    output_format: str = "safetensors"
    quantize: bool = False
    quantize_bits: Optional[int] = None


@dataclass
class Model:
    """Represents an uploaded model."""
    id: str
    name: str
    status: str
    size_mb: float
    format: str
    created_at: Optional[str] = None
    storage_path: Optional[str] = None
    
    def __post_init__(self):
        if isinstance(self.created_at, str):
            pass  # Keep as string for simplicity


@dataclass
class OptimizationResult:
    """Result of an optimization job."""
    job_id: str
    status: str
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    sparsity_achieved: float
    processing_time_seconds: float
    download_url: Optional[str] = None
    accuracy_metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def download(self, path: str) -> Path:
        """Download optimized model to path."""
        if not self.download_url:
            raise ValueError("No download URL available")
        
        path = Path(path)
        response = httpx.get(self.download_url, follow_redirects=True)
        response.raise_for_status()
        
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        
        return path


@dataclass
class OptimizationJob:
    """Represents an optimization job."""
    id: str
    status: str
    model_id: str
    config: Dict[str, Any]
    progress: float = 0.0
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    client: Optional[Any] = field(default=None, repr=False)
    
    @property
    def is_complete(self) -> bool:
        """Check if job is complete."""
        return self.status in ("completed", "failed", "cancelled")
    
    @property
    def is_running(self) -> bool:
        """Check if job is running."""
        return self.status == "running"
    
    def refresh(self) -> "OptimizationJob":
        """Refresh job status from API."""
        if not self.client:
            raise ValueError("Client not available")
        
        updated = self.client.get_job(self.id)
        self.status = updated.status
        self.progress = updated.progress
        self.result = updated.result
        self.error_message = updated.error_message
        self.completed_at = updated.completed_at
        
        return self
    
    def wait(self, poll_interval: float = 2.0, timeout: float = 3600) -> OptimizationResult:
        """
        Wait for job to complete.
        
        Args:
            poll_interval: Seconds between status checks
            timeout: Maximum wait time in seconds
        
        Returns:
            OptimizationResult when complete
        
        Raises:
            TimeoutError: If timeout exceeded
            OptimizationError: If job fails
        """
        import time
        from .exceptions import OptimizationError
        
        start_time = time.time()
        
        while not self.is_complete:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Job did not complete within {timeout}s")
            
            time.sleep(poll_interval)
            self.refresh()
        
        if self.status == "failed":
            raise OptimizationError(self.error_message or "Optimization failed")
        
        if self.status == "cancelled":
            raise OptimizationError("Job was cancelled")
        
        # Build result from job data
        result_data = self.result or {}
        return OptimizationResult(
            job_id=self.id,
            status=self.status,
            original_size_mb=result_data.get("original_size_mb", 0),
            optimized_size_mb=result_data.get("optimized_size_mb", 0),
            compression_ratio=result_data.get("compression_ratio", 1),
            sparsity_achieved=result_data.get("sparsity_achieved", 0),
            processing_time_seconds=result_data.get("processing_time_seconds", 0),
            download_url=result_data.get("download_url"),
            accuracy_metrics=result_data.get("accuracy_metrics"),
        )
    
    def cancel(self) -> bool:
        """Cancel the job."""
        if not self.client:
            raise ValueError("Client not available")
        
        self.client._request("POST", f"/optimize/{self.id}/cancel")
        self.status = "cancelled"
        return True


@dataclass
class UsageSummary:
    """User's usage summary."""
    tier: str
    models_used: int
    models_limit: int
    optimizations_used: int
    optimizations_limit: int
    storage_used_mb: float
    storage_limit_mb: float
    billing_period_start: Optional[str] = None
    billing_period_end: Optional[str] = None
    
    @property
    def models_remaining(self) -> int:
        """Models remaining in quota."""
        if self.models_limit == -1:
            return float("inf")
        return max(0, self.models_limit - self.models_used)
    
    @property
    def optimizations_remaining(self) -> int:
        """Optimizations remaining in quota."""
        if self.optimizations_limit == -1:
            return float("inf")
        return max(0, self.optimizations_limit - self.optimizations_used)
    
    @property
    def models_usage_percent(self) -> float:
        """Percentage of model quota used."""
        if self.models_limit <= 0:
            return 0
        return (self.models_used / self.models_limit) * 100
    
    @property
    def optimizations_usage_percent(self) -> float:
        """Percentage of optimization quota used."""
        if self.optimizations_limit <= 0:
            return 0
        return (self.optimizations_used / self.optimizations_limit) * 100
