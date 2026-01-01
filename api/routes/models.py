"""Model Management Routes."""

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field, ConfigDict

from api.routes.auth import CurrentUser, User
from api.config import settings

router = APIRouter()


# ============== Enums ==============

class ModelFormat(str, Enum):
    """Supported model formats."""
    PYTORCH = "pytorch"
    SAFETENSORS = "safetensors"
    ONNX = "onnx"


class ModelStatus(str, Enum):
    """Model processing status."""
    UPLOADING = "uploading"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DELETED = "deleted"


# ============== Schemas ==============

class ModelUploadRequest(BaseModel):
    """Request for presigned upload URL."""
    name: str = Field(..., min_length=1, max_length=100, description="Model name")
    format: ModelFormat = Field(..., description="Model file format")
    size_bytes: int = Field(..., gt=0, description="File size in bytes")


class ModelUploadResponse(BaseModel):
    """Response with presigned upload URL."""
    upload_url: str
    model_id: str
    expires_at: datetime


class ModelResponse(BaseModel):
    """Model information response."""
    id: str
    name: str
    format: ModelFormat
    size_mb: float
    status: ModelStatus
    storage_path: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)


class ModelList(BaseModel):
    """Paginated list of models."""
    models: list[ModelResponse]
    total: int
    page: int
    per_page: int
    has_next: bool


# ============== Fake Database (Replace with real DB) ==============

fake_models_db: dict = {}


class Model:
    """Model entity (placeholder)."""
    def __init__(
        self,
        id: str,
        user_id: str,
        name: str,
        format: ModelFormat,
        size_bytes: int,
        status: ModelStatus = ModelStatus.UPLOADING,
    ):
        self.id = id
        self.user_id = user_id
        self.name = name
        self.format = format
        self.size_bytes = size_bytes
        self.size_mb = size_bytes / (1024 * 1024)
        self.status = status
        self.storage_path: Optional[str] = None
        self.created_at = datetime.utcnow()
        self.updated_at: Optional[datetime] = None


# ============== Helper Functions ==============

def get_max_model_size(user: User) -> int:
    """Get maximum model size in MB for user's tier."""
    tier_limits = {
        "free": settings.max_model_size_free_mb,
        "pro": settings.max_model_size_pro_mb,
        "enterprise": settings.max_model_size_enterprise_mb,
    }
    return tier_limits.get(user.tier, settings.max_model_size_free_mb)


def generate_presigned_url(model_id: str, content_type: str) -> str:
    """Generate presigned S3 upload URL."""
    # TODO: Implement actual S3 presigned URL generation
    # For now, return placeholder
    return f"{settings.s3_endpoint}/{settings.s3_bucket}/{model_id}?presigned=true"


# ============== Endpoints ==============

@router.post("/upload-url", response_model=ModelUploadResponse)
async def get_upload_url(request: ModelUploadRequest, current_user: CurrentUser):
    """
    Get presigned URL for direct S3 upload.
    
    The client should upload the model file directly to the returned URL.
    """
    # Check file size limit
    size_mb = request.size_bytes / (1024 * 1024)
    max_size = get_max_model_size(current_user)
    
    if size_mb > max_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({size_mb:.1f}MB) exceeds limit ({max_size}MB) for {current_user.tier} tier"
        )
    
    # Create model record
    model_id = str(uuid.uuid4())
    model = Model(
        id=model_id,
        user_id=current_user.id,
        name=request.name,
        format=request.format,
        size_bytes=request.size_bytes,
    )
    fake_models_db[model_id] = model
    
    # Generate presigned URL
    content_type = {
        ModelFormat.PYTORCH: "application/octet-stream",
        ModelFormat.SAFETENSORS: "application/octet-stream",
        ModelFormat.ONNX: "application/octet-stream",
    }[request.format]
    
    upload_url = generate_presigned_url(model_id, content_type)
    expires_at = datetime.utcnow() + timedelta(hours=1)
    
    return ModelUploadResponse(
        upload_url=upload_url,
        model_id=model_id,
        expires_at=expires_at,
    )


@router.post("/{model_id}/confirm", response_model=ModelResponse)
async def confirm_upload(model_id: str, current_user: CurrentUser):
    """
    Confirm that model upload completed.
    
    Call this after successfully uploading to the presigned URL.
    """
    model = fake_models_db.get(model_id)
    
    if not model or model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    if model.status != ModelStatus.UPLOADING:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model is not in uploading state (current: {model.status})"
        )
    
    # TODO: Verify file exists in S3
    # s3_client.head_object(Bucket=bucket, Key=model_id)
    
    # Update model status
    model.status = ModelStatus.READY
    model.storage_path = f"s3://{settings.s3_bucket}/{model_id}"
    model.updated_at = datetime.utcnow()
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        format=model.format,
        size_mb=model.size_mb,
        status=model.status,
        storage_path=model.storage_path,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.get("", response_model=ModelList)
async def list_models(
    current_user: CurrentUser,
    page: Annotated[int, Query(ge=1)] = 1,
    per_page: Annotated[int, Query(ge=1, le=100)] = 20,
    status_filter: Optional[ModelStatus] = None,
    format_filter: Optional[ModelFormat] = None,
):
    """
    List user's models.
    
    Supports pagination and filtering by status and format.
    """
    # Get user's models
    user_models = [
        m for m in fake_models_db.values()
        if m.user_id == current_user.id and m.status != ModelStatus.DELETED
    ]
    
    # Apply filters
    if status_filter:
        user_models = [m for m in user_models if m.status == status_filter]
    if format_filter:
        user_models = [m for m in user_models if m.format == format_filter]
    
    # Sort by created_at descending
    user_models.sort(key=lambda x: x.created_at, reverse=True)
    
    # Paginate
    total = len(user_models)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_models = user_models[start:end]
    
    return ModelList(
        models=[
            ModelResponse(
                id=m.id,
                name=m.name,
                format=m.format,
                size_mb=m.size_mb,
                status=m.status,
                storage_path=m.storage_path,
                created_at=m.created_at,
                updated_at=m.updated_at,
            )
            for m in paginated_models
        ],
        total=total,
        page=page,
        per_page=per_page,
        has_next=end < total,
    )


@router.get("/{model_id}", response_model=ModelResponse)
async def get_model(model_id: str, current_user: CurrentUser):
    """Get model by ID."""
    model = fake_models_db.get(model_id)
    
    if not model or model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    return ModelResponse(
        id=model.id,
        name=model.name,
        format=model.format,
        size_mb=model.size_mb,
        status=model.status,
        storage_path=model.storage_path,
        created_at=model.created_at,
        updated_at=model.updated_at,
    )


@router.delete("/{model_id}")
async def delete_model(model_id: str, current_user: CurrentUser):
    """
    Delete a model.
    
    This soft-deletes the model and schedules S3 cleanup.
    """
    model = fake_models_db.get(model_id)
    
    if not model or model.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found"
        )
    
    # Soft delete
    model.status = ModelStatus.DELETED
    model.updated_at = datetime.utcnow()
    
    # TODO: Queue S3 cleanup task
    # cleanup_model_files.delay(model_id)
    
    return {"message": "Model deleted", "model_id": model_id}
