"""Billing and Usage Routes."""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.routes.auth import CurrentUser
from api.config import settings

router = APIRouter()


# ============== Schemas ==============

class TierLimits(BaseModel):
    """Limits for a pricing tier."""
    models_per_month: int
    optimizations_per_month: int
    max_model_size_mb: int
    storage_gb: int


class PricingTier(BaseModel):
    """Pricing tier information."""
    id: str
    name: str
    price_monthly: float
    price_annual: float
    features: list[str]
    limits: TierLimits
    cta: str
    popular: bool = False


class PricingResponse(BaseModel):
    """Pricing information response."""
    tiers: list[PricingTier]


class UsageMetric(BaseModel):
    """Single usage metric."""
    used: int
    limit: int
    percentage: float


class UsageSummary(BaseModel):
    """User's usage summary."""
    tier: str
    billing_period_start: datetime
    billing_period_end: datetime
    models: UsageMetric
    optimizations: UsageMetric
    storage_mb: UsageMetric


class CheckoutRequest(BaseModel):
    """Checkout session request."""
    tier: str = Field(..., pattern="^(pro|enterprise)$")
    annual: bool = False


class CheckoutResponse(BaseModel):
    """Checkout session response."""
    checkout_url: str


class SubscriptionInfo(BaseModel):
    """Current subscription information."""
    tier: str
    status: str
    current_period_start: datetime
    current_period_end: datetime
    cancel_at_period_end: bool


# ============== Pricing Data ==============

PRICING_TIERS = [
    PricingTier(
        id="free",
        name="Free",
        price_monthly=0,
        price_annual=0,
        features=[
            "3 models per month",
            "5 optimizations per month",
            "500MB max model size",
            "1GB storage",
            "Community support",
        ],
        limits=TierLimits(
            models_per_month=3,
            optimizations_per_month=5,
            max_model_size_mb=500,
            storage_gb=1,
        ),
        cta="Get Started",
    ),
    PricingTier(
        id="pro",
        name="Pro",
        price_monthly=49,
        price_annual=470,
        features=[
            "50 models per month",
            "100 optimizations per month",
            "5GB max model size",
            "50GB storage",
            "Priority support",
            "Webhooks",
            "API rate limit: 60/min",
        ],
        limits=TierLimits(
            models_per_month=50,
            optimizations_per_month=100,
            max_model_size_mb=5000,
            storage_gb=50,
        ),
        cta="Upgrade to Pro",
        popular=True,
    ),
    PricingTier(
        id="enterprise",
        name="Enterprise",
        price_monthly=499,
        price_annual=4790,
        features=[
            "Unlimited models",
            "Unlimited optimizations",
            "50GB max model size",
            "500GB storage",
            "Dedicated support",
            "SLA guarantee",
            "Custom integrations",
            "API rate limit: 300/min",
        ],
        limits=TierLimits(
            models_per_month=-1,
            optimizations_per_month=-1,
            max_model_size_mb=50000,
            storage_gb=500,
        ),
        cta="Contact Sales",
    ),
]


# ============== Fake Usage Tracking ==============

fake_usage_db: dict = {}


def get_or_create_usage(user_id: str) -> dict:
    """Get or create usage record for user."""
    if user_id not in fake_usage_db:
        fake_usage_db[user_id] = {
            "models": 0,
            "optimizations": 0,
            "storage_mb": 0,
            "period_start": datetime.utcnow(),
        }
    return fake_usage_db[user_id]


# ============== Endpoints ==============

@router.get("/pricing", response_model=PricingResponse)
async def get_pricing():
    """
    Get pricing information.
    
    This endpoint is public (no authentication required).
    """
    return PricingResponse(tiers=PRICING_TIERS)


@router.get("/usage", response_model=UsageSummary)
async def get_usage(current_user: CurrentUser):
    """Get current usage for billing period."""
    usage = get_or_create_usage(current_user.id)
    
    # Get tier limits
    tier_data = next((t for t in PRICING_TIERS if t.id == current_user.tier), PRICING_TIERS[0])
    limits = tier_data.limits
    
    # Calculate usage percentages
    models_limit = limits.models_per_month if limits.models_per_month > 0 else 999999
    opt_limit = limits.optimizations_per_month if limits.optimizations_per_month > 0 else 999999
    storage_limit_mb = limits.storage_gb * 1024
    
    return UsageSummary(
        tier=current_user.tier,
        billing_period_start=usage["period_start"],
        billing_period_end=datetime.utcnow(),  # Placeholder
        models=UsageMetric(
            used=usage["models"],
            limit=models_limit,
            percentage=(usage["models"] / models_limit) * 100 if models_limit > 0 else 0,
        ),
        optimizations=UsageMetric(
            used=usage["optimizations"],
            limit=opt_limit,
            percentage=(usage["optimizations"] / opt_limit) * 100 if opt_limit > 0 else 0,
        ),
        storage_mb=UsageMetric(
            used=usage["storage_mb"],
            limit=storage_limit_mb,
            percentage=(usage["storage_mb"] / storage_limit_mb) * 100,
        ),
    )


@router.post("/checkout", response_model=CheckoutResponse)
async def create_checkout(request: CheckoutRequest, current_user: CurrentUser):
    """
    Create Stripe checkout session for upgrading.
    
    Returns a URL to redirect the user to Stripe's checkout page.
    """
    if current_user.tier == request.tier:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Already on {request.tier} tier"
        )
    
    # TODO: Create actual Stripe checkout session
    # session = stripe.checkout.Session.create(...)
    
    # Placeholder URL
    checkout_url = f"https://checkout.stripe.com/pay/fake_session_{request.tier}"
    
    return CheckoutResponse(checkout_url=checkout_url)


@router.post("/portal", response_model=CheckoutResponse)
async def create_portal(current_user: CurrentUser):
    """
    Create Stripe customer portal session.
    
    Allows users to manage their subscription, update payment method, etc.
    """
    if current_user.tier == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to manage"
        )
    
    # TODO: Create actual Stripe portal session
    # session = stripe.billing_portal.Session.create(...)
    
    portal_url = "https://billing.stripe.com/p/fake_portal"
    
    return CheckoutResponse(checkout_url=portal_url)


@router.get("/subscription", response_model=Optional[SubscriptionInfo])
async def get_subscription(current_user: CurrentUser):
    """Get current subscription information."""
    if current_user.tier == "free":
        return None
    
    # TODO: Get actual subscription from Stripe
    return SubscriptionInfo(
        tier=current_user.tier,
        status="active",
        current_period_start=datetime.utcnow(),
        current_period_end=datetime.utcnow(),
        cancel_at_period_end=False,
    )


@router.post("/cancel")
async def cancel_subscription(current_user: CurrentUser):
    """
    Cancel subscription at end of billing period.
    
    User keeps access until the period ends.
    """
    if current_user.tier == "free":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No active subscription to cancel"
        )
    
    # TODO: Cancel in Stripe
    # stripe.Subscription.modify(sub_id, cancel_at_period_end=True)
    
    return {"message": "Subscription will be cancelled at end of billing period"}


@router.post("/webhook")
async def stripe_webhook():
    """
    Handle Stripe webhook events.
    
    Verifies signature and processes events.
    """
    # TODO: Implement actual webhook handling
    # event = stripe.Webhook.construct_event(payload, sig, secret)
    
    return {"received": True}
