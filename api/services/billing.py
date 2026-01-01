"""
Billing & Usage Tracking Service
================================

Tracks usage and manages quotas per user tier.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from api.config import settings
from api.services.cache import get_cache_service


class UserTier(str, Enum):
    """User subscription tiers."""
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


@dataclass
class TierLimits:
    """Limits for a subscription tier."""
    models_per_month: int
    optimizations_per_month: int
    max_model_size_mb: int
    storage_gb: int
    api_rate_limit: int  # per minute


# Tier configurations
TIER_LIMITS = {
    UserTier.FREE: TierLimits(
        models_per_month=3,
        optimizations_per_month=5,
        max_model_size_mb=500,
        storage_gb=1,
        api_rate_limit=10,
    ),
    UserTier.PRO: TierLimits(
        models_per_month=50,
        optimizations_per_month=100,
        max_model_size_mb=5000,
        storage_gb=50,
        api_rate_limit=60,
    ),
    UserTier.ENTERPRISE: TierLimits(
        models_per_month=-1,  # Unlimited
        optimizations_per_month=-1,
        max_model_size_mb=50000,
        storage_gb=500,
        api_rate_limit=300,
    ),
}


@dataclass
class QuotaCheck:
    """Result of quota check."""
    allowed: bool
    current_usage: int
    limit: int
    percentage: float
    message: Optional[str] = None


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
    billing_period_start: datetime
    billing_period_end: datetime


class BillingService:
    """
    Service for tracking usage and managing quotas.
    """
    
    def __init__(self):
        self.cache = get_cache_service()
    
    def get_tier_limits(self, tier: str) -> TierLimits:
        """Get limits for a tier."""
        try:
            tier_enum = UserTier(tier)
            return TIER_LIMITS.get(tier_enum, TIER_LIMITS[UserTier.FREE])
        except ValueError:
            return TIER_LIMITS[UserTier.FREE]
    
    async def check_quota(
        self,
        user_id: str,
        tier: str,
        action: str
    ) -> QuotaCheck:
        """
        Check if user can perform action.
        
        Args:
            user_id: User identifier
            tier: User's subscription tier
            action: Action type (model_upload, optimization, etc.)
        
        Returns:
            QuotaCheck with allowed status and usage info
        """
        limits = self.get_tier_limits(tier)
        
        # Get current usage from cache or compute
        usage_key = f"usage:{user_id}:{self._get_period_key()}"
        usage = await self.cache.get(usage_key) or {}
        
        if action == "model_upload":
            current = usage.get("models", 0)
            limit = limits.models_per_month
            
        elif action == "optimization":
            current = usage.get("optimizations", 0)
            limit = limits.optimizations_per_month
            
        else:
            return QuotaCheck(
                allowed=True,
                current_usage=0,
                limit=-1,
                percentage=0,
            )
        
        # Unlimited check
        if limit == -1:
            return QuotaCheck(
                allowed=True,
                current_usage=current,
                limit=limit,
                percentage=0,
                message="Unlimited",
            )
        
        # Check if over limit
        if current >= limit:
            return QuotaCheck(
                allowed=False,
                current_usage=current,
                limit=limit,
                percentage=100.0,
                message=f"Monthly {action} limit reached ({limit}). Upgrade to increase limits.",
            )
        
        percentage = (current / limit * 100) if limit > 0 else 0
        
        return QuotaCheck(
            allowed=True,
            current_usage=current,
            limit=limit,
            percentage=percentage,
        )
    
    async def record_usage(
        self,
        user_id: str,
        action: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Record usage event.
        
        Args:
            user_id: User identifier
            action: Action type
            details: Additional details (size, duration, etc.)
        """
        period_key = self._get_period_key()
        usage_key = f"usage:{user_id}:{period_key}"
        
        # Get current usage
        usage = await self.cache.get(usage_key) or {}
        
        # Increment counter
        if action == "model_upload":
            usage["models"] = usage.get("models", 0) + 1
            if details and "size_mb" in details:
                usage["storage_mb"] = usage.get("storage_mb", 0) + details["size_mb"]
                
        elif action == "optimization":
            usage["optimizations"] = usage.get("optimizations", 0) + 1
            if details and "compute_seconds" in details:
                usage["compute_seconds"] = usage.get("compute_seconds", 0) + details["compute_seconds"]
        
        # Store with TTL (expire at end of month + buffer)
        ttl = self._seconds_until_month_end() + 86400  # +1 day buffer
        await self.cache.set(usage_key, usage, ttl=ttl)
    
    async def get_usage_summary(self, user_id: str, tier: str) -> UsageSummary:
        """
        Get usage summary for user.
        
        Args:
            user_id: User identifier
            tier: User's subscription tier
        
        Returns:
            UsageSummary with current usage
        """
        limits = self.get_tier_limits(tier)
        period_key = self._get_period_key()
        usage_key = f"usage:{user_id}:{period_key}"
        
        usage = await self.cache.get(usage_key) or {}
        
        # Calculate billing period
        now = datetime.utcnow()
        period_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        if now.month == 12:
            period_end = period_start.replace(year=now.year + 1, month=1)
        else:
            period_end = period_start.replace(month=now.month + 1)
        
        return UsageSummary(
            tier=tier,
            models_used=usage.get("models", 0),
            models_limit=limits.models_per_month,
            optimizations_used=usage.get("optimizations", 0),
            optimizations_limit=limits.optimizations_per_month,
            storage_used_mb=usage.get("storage_mb", 0),
            storage_limit_mb=limits.storage_gb * 1024,
            billing_period_start=period_start,
            billing_period_end=period_end,
        )
    
    def _get_period_key(self) -> str:
        """Get current billing period key (YYYY-MM)."""
        return datetime.utcnow().strftime("%Y-%m")
    
    def _seconds_until_month_end(self) -> int:
        """Calculate seconds until end of current month."""
        now = datetime.utcnow()
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        
        return int((next_month - now).total_seconds())


# Singleton instance
_billing_service: Optional[BillingService] = None


def get_billing_service() -> BillingService:
    """Get or create billing service singleton."""
    global _billing_service
    if _billing_service is None:
        _billing_service = BillingService()
    return _billing_service
