"""Webhook Management Routes."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid
import hmac
import hashlib

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field, HttpUrl

from api.routes.auth import CurrentUser

router = APIRouter()


# ============== Enums ==============

class WebhookEvent(str, Enum):
    """Supported webhook events."""
    MODEL_UPLOADED = "model.uploaded"
    MODEL_DELETED = "model.deleted"
    JOB_STARTED = "job.started"
    JOB_COMPLETED = "job.completed"
    JOB_FAILED = "job.failed"


# ============== Schemas ==============

class WebhookCreate(BaseModel):
    """Create webhook request."""
    url: HttpUrl = Field(..., description="Webhook endpoint URL")
    events: list[WebhookEvent] = Field(
        ..., 
        min_length=1,
        description="Events to subscribe to"
    )
    secret: str = Field(
        ..., 
        min_length=16,
        description="Secret for signing webhook payloads"
    )


class WebhookResponse(BaseModel):
    """Webhook information response."""
    id: str
    url: str
    events: list[WebhookEvent]
    is_active: bool
    created_at: datetime
    last_triggered_at: Optional[datetime] = None
    success_count: int = 0
    failure_count: int = 0


class WebhookTestResponse(BaseModel):
    """Webhook test result."""
    success: bool
    status_code: Optional[int] = None
    response_time_ms: Optional[float] = None
    error: Optional[str] = None


# ============== Fake Database ==============

fake_webhooks_db: dict = {}


class Webhook:
    """Webhook entity."""
    def __init__(
        self,
        id: str,
        user_id: str,
        url: str,
        events: list[WebhookEvent],
        secret: str,
    ):
        self.id = id
        self.user_id = user_id
        self.url = url
        self.events = events
        self.secret = secret
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.last_triggered_at: Optional[datetime] = None
        self.success_count = 0
        self.failure_count = 0


# ============== Helper Functions ==============

def sign_payload(payload: str, secret: str) -> str:
    """Sign payload with HMAC-SHA256."""
    return hmac.new(
        secret.encode(),
        payload.encode(),
        hashlib.sha256
    ).hexdigest()


async def send_webhook(webhook: Webhook, event: str, data: dict) -> bool:
    """Send webhook request."""
    import httpx
    import json
    import time
    
    payload = json.dumps({
        "event": event,
        "timestamp": datetime.utcnow().isoformat(),
        "data": data,
    })
    
    signature = sign_payload(payload, webhook.secret)
    
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook.url,
                content=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": f"sha256={signature}",
                    "X-Webhook-Event": event,
                },
                timeout=10.0,
            )
        
        webhook.last_triggered_at = datetime.utcnow()
        
        if response.is_success:
            webhook.success_count += 1
            return True
        else:
            webhook.failure_count += 1
            return False
            
    except Exception:
        webhook.failure_count += 1
        return False


# ============== Endpoints ==============

@router.post("", response_model=WebhookResponse, status_code=status.HTTP_201_CREATED)
async def create_webhook(request: WebhookCreate, current_user: CurrentUser):
    """
    Register a new webhook endpoint.
    
    Webhooks will receive POST requests with a signed payload.
    Verify the X-Webhook-Signature header using HMAC-SHA256.
    """
    # Check limit (max 5 webhooks per user)
    user_webhooks = [w for w in fake_webhooks_db.values() if w.user_id == current_user.id]
    if len(user_webhooks) >= 5:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum number of webhooks (5) reached"
        )
    
    webhook_id = str(uuid.uuid4())
    webhook = Webhook(
        id=webhook_id,
        user_id=current_user.id,
        url=str(request.url),
        events=request.events,
        secret=request.secret,
    )
    fake_webhooks_db[webhook_id] = webhook
    
    return WebhookResponse(
        id=webhook.id,
        url=webhook.url,
        events=webhook.events,
        is_active=webhook.is_active,
        created_at=webhook.created_at,
        last_triggered_at=webhook.last_triggered_at,
        success_count=webhook.success_count,
        failure_count=webhook.failure_count,
    )


@router.get("", response_model=list[WebhookResponse])
async def list_webhooks(current_user: CurrentUser):
    """List all webhooks for current user."""
    user_webhooks = [w for w in fake_webhooks_db.values() if w.user_id == current_user.id]
    
    return [
        WebhookResponse(
            id=w.id,
            url=w.url,
            events=w.events,
            is_active=w.is_active,
            created_at=w.created_at,
            last_triggered_at=w.last_triggered_at,
            success_count=w.success_count,
            failure_count=w.failure_count,
        )
        for w in user_webhooks
    ]


@router.get("/{webhook_id}", response_model=WebhookResponse)
async def get_webhook(webhook_id: str, current_user: CurrentUser):
    """Get webhook by ID."""
    webhook = fake_webhooks_db.get(webhook_id)
    
    if not webhook or webhook.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    return WebhookResponse(
        id=webhook.id,
        url=webhook.url,
        events=webhook.events,
        is_active=webhook.is_active,
        created_at=webhook.created_at,
        last_triggered_at=webhook.last_triggered_at,
        success_count=webhook.success_count,
        failure_count=webhook.failure_count,
    )


@router.delete("/{webhook_id}")
async def delete_webhook(webhook_id: str, current_user: CurrentUser):
    """Delete a webhook."""
    webhook = fake_webhooks_db.get(webhook_id)
    
    if not webhook or webhook.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    del fake_webhooks_db[webhook_id]
    
    return {"message": "Webhook deleted", "webhook_id": webhook_id}


@router.post("/{webhook_id}/test", response_model=WebhookTestResponse)
async def test_webhook(webhook_id: str, current_user: CurrentUser):
    """
    Send a test event to the webhook.
    
    Sends a test.ping event to verify the webhook is working.
    """
    import httpx
    import json
    import time
    
    webhook = fake_webhooks_db.get(webhook_id)
    
    if not webhook or webhook.user_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Webhook not found"
        )
    
    payload = json.dumps({
        "event": "test.ping",
        "timestamp": datetime.utcnow().isoformat(),
        "data": {"message": "This is a test webhook"},
    })
    
    signature = sign_payload(payload, webhook.secret)
    
    try:
        start = time.perf_counter()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                webhook.url,
                content=payload,
                headers={
                    "Content-Type": "application/json",
                    "X-Webhook-Signature": f"sha256={signature}",
                    "X-Webhook-Event": "test.ping",
                },
                timeout=10.0,
            )
        elapsed = (time.perf_counter() - start) * 1000
        
        return WebhookTestResponse(
            success=response.is_success,
            status_code=response.status_code,
            response_time_ms=round(elapsed, 2),
        )
        
    except httpx.TimeoutException:
        return WebhookTestResponse(
            success=False,
            error="Request timed out after 10 seconds"
        )
    except httpx.RequestError as e:
        return WebhookTestResponse(
            success=False,
            error=str(e)
        )
