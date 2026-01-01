"""Health Check Routes."""

from datetime import datetime
from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ComponentHealth(BaseModel):
    """Health status of a single component."""
    status: Literal["ok", "degraded", "error"]
    latency_ms: Optional[float] = None
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Overall health response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    components: Dict[str, ComponentHealth]
    timestamp: datetime


@router.get("", response_model=Dict[str, Any])
async def health_check():
    """
    Basic health check.
    
    Returns simple status for load balancers.
    """
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
    }


@router.get("/ready", response_model=HealthResponse)
async def readiness_check():
    """
    Detailed readiness check.
    
    Checks all dependencies:
    - Database connection
    - Redis connection
    - Celery workers
    
    Returns detailed status for monitoring.
    """
    import time
    
    components: Dict[str, ComponentHealth] = {}
    overall_status: Literal["healthy", "degraded", "unhealthy"] = "healthy"
    
    # Check database
    try:
        start = time.perf_counter()
        # TODO: Actual database ping
        # await database.execute("SELECT 1")
        latency = (time.perf_counter() - start) * 1000
        components["database"] = ComponentHealth(status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        components["database"] = ComponentHealth(status="error", message=str(e))
        overall_status = "unhealthy"
    
    # Check Redis
    try:
        start = time.perf_counter()
        # TODO: Actual Redis ping
        # await redis.ping()
        latency = (time.perf_counter() - start) * 1000
        components["redis"] = ComponentHealth(status="ok", latency_ms=round(latency, 2))
    except Exception as e:
        components["redis"] = ComponentHealth(status="error", message=str(e))
        if overall_status != "unhealthy":
            overall_status = "degraded"
    
    # Check Celery workers
    try:
        # TODO: Actual Celery inspection
        # inspect = celery_app.control.inspect()
        # workers = inspect.active()
        workers_count = 0  # Placeholder
        if workers_count > 0:
            components["celery"] = ComponentHealth(
                status="ok", 
                message=f"{workers_count} workers active"
            )
        else:
            components["celery"] = ComponentHealth(
                status="degraded",
                message="No workers detected"
            )
            if overall_status == "healthy":
                overall_status = "degraded"
    except Exception as e:
        components["celery"] = ComponentHealth(status="error", message=str(e))
        if overall_status == "healthy":
            overall_status = "degraded"
    
    return HealthResponse(
        status=overall_status,
        components=components,
        timestamp=datetime.utcnow(),
    )


@router.get("/live")
async def liveness_check():
    """
    Simple liveness probe.
    
    Always returns 200 if the application is running.
    Used by Kubernetes liveness probes.
    """
    return {"status": "alive"}


@router.get("/startup")
async def startup_check():
    """
    Startup probe.
    
    Returns 200 once the application has completed startup.
    """
    # TODO: Check if all startup tasks are complete
    return {
        "status": "started",
        "timestamp": datetime.utcnow().isoformat(),
    }
