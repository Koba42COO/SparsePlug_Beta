"""
Rate Limiting Middleware
========================

Implements per-user rate limiting based on subscription tier.
"""

from datetime import datetime
from typing import Optional, Callable, Dict

from fastapi import Request, HTTPException, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.config import settings
from api.services.cache import get_cache_service


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware.
    
    Limits requests based on:
    - User tier (free/pro/enterprise)
    - Endpoint type (read/write/optimize)
    - Time window (per minute)
    """
    
    # Rate limits per tier (requests per minute)
    TIER_LIMITS = {
        "free": {
            "default": 10,
            "read": 30,
            "write": 10,
            "optimize": 3,
        },
        "pro": {
            "default": 60,
            "read": 120,
            "write": 60,
            "optimize": 20,
        },
        "enterprise": {
            "default": 300,
            "read": 600,
            "write": 300,
            "optimize": 100,
        },
    }
    
    # Endpoint type mapping
    ENDPOINT_TYPES = {
        "/health": "read",
        "/billing/pricing": "read",
        "/models": "read",
        "/optimize": "read",
        "/models/upload-url": "write",
        "/optimize": "optimize",
    }
    
    # Endpoints exempt from rate limiting
    EXEMPT_ENDPOINTS = {
        "/",
        "/version",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/health",
        "/health/live",
        "/health/ready",
    }
    
    async def dispatch(self, request: Request, call_next: Callable):
        """Process request with rate limiting."""
        
        # Skip rate limiting for exempt endpoints
        path = request.url.path
        if path in self.EXEMPT_ENDPOINTS:
            return await call_next(request)
        
        # Skip rate limiting for OPTIONS (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)
        
        # Get user info from request state (set by auth)
        user_id = getattr(request.state, "user_id", None)
        user_tier = getattr(request.state, "user_tier", "free")
        
        # If no user, use IP-based limiting
        if not user_id:
            user_id = request.client.host if request.client else "anonymous"
            user_tier = "free"
        
        # Determine endpoint type
        endpoint_type = self._get_endpoint_type(path, request.method)
        
        # Get rate limit for tier and endpoint
        limit = self._get_rate_limit(user_tier, endpoint_type)
        
        # Check rate limit
        cache = get_cache_service()
        rate_key = f"rate:{user_id}:{endpoint_type}"
        
        try:
            allowed, remaining, reset = await cache.rate_limit_check(
                rate_key, limit, 60  # 60 second window
            )
        except Exception:
            # Fail open if cache is unavailable
            allowed, remaining, reset = True, limit, 60
        
        # Add rate limit headers
        response = await call_next(request) if allowed else JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={
                "error": {
                    "code": "RATE_LIMITED",
                    "message": f"Rate limit exceeded. Try again in {reset} seconds.",
                },
            },
        )
        
        response.headers["X-RateLimit-Limit"] = str(limit)
        response.headers["X-RateLimit-Remaining"] = str(max(0, remaining))
        response.headers["X-RateLimit-Reset"] = str(reset)
        
        return response
    
    def _get_endpoint_type(self, path: str, method: str) -> str:
        """Determine endpoint type for rate limiting."""
        # Check specific paths
        for endpoint, type_ in self.ENDPOINT_TYPES.items():
            if path.startswith(endpoint):
                return type_
        
        # Fall back to method-based detection
        if method == "GET":
            return "read"
        elif method in ("POST", "PUT", "PATCH", "DELETE"):
            return "write"
        
        return "default"
    
    def _get_rate_limit(self, tier: str, endpoint_type: str) -> int:
        """Get rate limit for tier and endpoint type."""
        tier_limits = self.TIER_LIMITS.get(tier, self.TIER_LIMITS["free"])
        return tier_limits.get(endpoint_type, tier_limits["default"])


def get_rate_limit_headers(user_tier: str = "free") -> Dict[str, str]:
    """Get rate limit info for documentation."""
    limits = RateLimitMiddleware.TIER_LIMITS.get(user_tier, {})
    return {
        "X-RateLimit-Limit": f"{limits.get('default', 10)}/minute",
        "X-RateLimit-Remaining": "varies",
        "X-RateLimit-Reset": "seconds until reset",
    }
