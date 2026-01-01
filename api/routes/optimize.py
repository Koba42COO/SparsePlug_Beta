"""Optimization Job Routes."""

from datetime import datetime
from enum import Enum
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, ConfigDict

from api.routes.auth import CurrentUser, User
from api.routes.models import fake_models_db, ModelStatus, ModelFormat
from api.config import settings

router = APIRouter()


# ============== Enums ==============

class JobStatus(str, Enum):
    """Optimization job status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OutputFormat(str, Enum):
    """Output model format."""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"


# ============== Schemas ==============

class OptimizationConfig(BaseModel):
    """Configuration for optimization job."""
    target_sparsity: float = Field(
        default=0.96,
        ge=0.0,
        le=0.99,
        description="Target sparsity level (0.0-0.99). Default is 96%."
    )
    preserve_accuracy: bool = Field(
        default=True,
        description="Use accuracy-preserving optimization"
    )
    output_format: OutputFormat = Field(
        default=OutputFormat.SAFETENSORS,
        description="Format for optimized model"
    )
    quantize: bool = Field(
        default=False,
        description="Apply INT8 quantization after sparsification"
    )
    quantize_bits: Optional[int] = Field(
        default=None,
        description="Quantization bits (4 or 8). Only used if quantize=True"
    )


class JobCreate(BaseModel):
    """Create optimization job request."""
    model_id: str = Field(..., description="ID of the model to optimize")
    config: OptimizationConfig = Field(default_factory=OptimizationConfig)


class JobResult(BaseModel):
    """Optimization job results."""
    original_size_mb: float
    optimized_size_mb: float
    compression_ratio: float
    sparsity_achieved: float
    accuracy_metrics: dict
    download_url: str
    expires_at: datetime


class JobResponse(BaseModel):
    """Job information response."""
    id: str
    model_id: str
    status: JobStatus
    config: OptimizationConfig
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    result: Optional[JobResult] = None
    error_message: Optional[str] = None
    estimated_time_seconds: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class JobList(BaseModel):
    """Paginated list of jobs."""
    jobs: list[JobResponse]
    total: int
    page: int
    per_page: int


# ============== Fake Database (Replace with real DB) ==============

fake_jobs_db: dict = {}


class Job:
    """Job entity (placeholder)."""
    def __init__(
        self,
        id: str,
        user_id: str,
        model_id: str,
        config: OptimizationConfig,
    ):
        self.id = id
        self.user_id = user_id
        self.model_id = model_id
        self.config = config
        self.status = JobStatus.QUEUED
        self.progress = 0.0
        self.result: Optional[JobResult] = None
        self.error_message: Optional[str] = None
        self.estimated_time_seconds: Optional[int] = None
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None


# ============== Helper Functions ==============

def estimate_processing_time(model_size_mb: float, config: OptimizationConfig) -> int:
    """Estimate processing time in seconds."""
    # Base time: 10 seconds per 100MB
    base_time = (model_size_mb / 100) * 10
    
    # Add time for higher sparsity
    sparsity_factor = 1 + (config.target_sparsity * 0.5)
    
    # Add time for quantization
    quantize_factor = 1.5 if config.quantize else 1.0
    
    return int(base_time * sparsity_factor * quantize_factor)


# ============== Endpoints ==============

@router.post("", response_model=JobResponse, status_code=status.HTTP_201_CREATED)
async def create_job(request: JobCreate, current_user: CurrentUser):
    """
    Create a new optimization job.
    
    The job will be queued and processed by a Celery worker.
    """
    # Verify model exists and belongs to user
    model = fake_models_db.get(request.model_id)
    
    if not model or model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.status != ModelStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not ready (status: {model.status})"
        )
    
    # TODO: Check user quota
    # quota = await check_quota(current_user, "optimization")
    # if not quota.allowed:
    #     raise HTTPException(status_code=403, detail="Quota exceeded")
    
    # Create job
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        user_id=current_user.id,
        model_id=request.model_id,
        config=request.config,
    )
    job.estimated_time_seconds = estimate_processing_time(model.size_mb, request.config)
    
    fake_jobs_db[job_id] = job
    
    # TODO: Queue Celery task
    # optimize_model.delay(job_id)
    
    return JobResponse(
        id=job.id,
        model_id=job.model_id,
        status=job.status,
        config=job.config,
        progress=job.progress,
        result=job.result,
        error_message=job.error_message,
        estimated_time_seconds=job.estimated_time_seconds,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.get("", response_model=JobList)
async def list_jobs(
    current_user: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: Optional[JobStatus] = None,
    model_id: Optional[str] = None,
):
    """
    List user's optimization jobs.
    
    Supports filtering by status and model.
    """
    # Get user's jobs
    user_jobs = [j for j in fake_jobs_db.values() if j.user_id == current_user.id]
    
    # Apply filters
    if status_filter:
        user_jobs = [j for j in user_jobs if j.status == status_filter]
    if model_id:
        user_jobs = [j for j in user_jobs if j.model_id == model_id]
    
    # Sort by created_at descending
    user_jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    total = len(user_jobs)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_jobs = user_jobs[start:end]
    
    return JobList(
        jobs=[
            JobResponse(
                id=j.id,
                model_id=j.model_id,
                status=j.status,
                config=j.config,
                progress=j.progress,
                result=j.result,
                error_message=j.error_message,
                estimated_time_seconds=j.estimated_time_seconds,
                created_at=j.created_at,
                started_at=j.started_at,
                completed_at=j.completed_at,
            )
            for j in paginated_jobs
        ],
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, current_user: CurrentUser):
    """Get job by ID with current status and results."""
    job = fake_jobs_db.get(job_id)
    
    if not job or job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    return JobResponse(
        id=job.id,
        model_id=job.model_id,
        status=job.status,
        config=job.config,
        progress=job.progress,
        result=job.result,
        error_message=job.error_message,
        estimated_time_seconds=job.estimated_time_seconds,
        created_at=job.created_at,
        started_at=job.started_at,
        completed_at=job.completed_at,
    )


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str, current_user: CurrentUser):
    """
    Cancel a queued or running job.
    
    Running jobs will be stopped as soon as possible.
    """
    job = fake_jobs_db.get(job_id)
    
    if not job or job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    if job.status not in [JobStatus.QUEUED, JobStatus.RUNNING]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Cannot cancel job with status: {job.status}"
        )
    
    # TODO: Cancel Celery task
    # celery_app.control.revoke(job_id, terminate=True)
    
    job.status = JobStatus.CANCELLED
    job.completed_at = datetime.utcnow()
    
    return {"message": "Job cancelled", "job_id": job_id}


@router.get("/{job_id}/logs")
async def get_job_logs(job_id: str, current_user: CurrentUser):
    """
    Stream job logs via Server-Sent Events.
    
    Useful for monitoring job progress in real-time.
    """
    job = fake_jobs_db.get(job_id)
    
    if not job or job.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Job not found"
        )
    
    async def generate_logs():
        """Generate SSE log stream."""
        # TODO: Implement actual log streaming from Redis pub/sub
        import asyncio
        
        # Send initial status
        yield f"data: {{'status': '{job.status}', 'progress': {job.progress}}}\n\n"
        
        # Simulate log updates
        for i in range(5):
            await asyncio.sleep(1)
            yield f"data: {{'message': 'Processing step {i+1}/5'}}\n\n"
        
        yield f"data: {{'status': 'complete'}}\n\n"
    
    return StreamingResponse(
        generate_logs(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
