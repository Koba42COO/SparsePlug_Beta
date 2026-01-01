"""
Redis Caching Service
=====================

Provides caching layer for frequently accessed data.
"""

import json
from typing import Optional, Any, TypeVar, Callable
from datetime import timedelta
import hashlib

import redis.asyncio as redis

from api.config import settings


T = TypeVar('T')


class CacheService:
    """
    Redis-based caching service.
    
    Features:
    - Async operations
    - JSON serialization
    - TTL support
    - Cache invalidation
    - Distributed locking
    """
    
    # Default TTLs
    TTL_SHORT = 60  # 1 minute
    TTL_MEDIUM = 300  # 5 minutes
    TTL_LONG = 3600  # 1 hour
    TTL_DAY = 86400  # 24 hours
    
    def __init__(self):
        self.redis: Optional[redis.Redis] = None
        self._prefix = "ps:"  # Prime-Sparse prefix
    
    async def connect(self):
        """Connect to Redis."""
        if self.redis is None:
            self.redis = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
            )
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self.redis:
            await self.redis.close()
            self.redis = None
    
    def _make_key(self, key: str) -> str:
        """Create namespaced key."""
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
        
        Returns:
            Cached value or None
        """
        await self.connect()
        
        try:
            value = await self.redis.get(self._make_key(key))
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            print(f"Cache get error: {e}")
            return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = TTL_MEDIUM
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds
        
        Returns:
            True if successful
        """
        await self.connect()
        
        try:
            await self.redis.setex(
                self._make_key(key),
                ttl,
                json.dumps(value),
            )
            return True
        except Exception as e:
            print(f"Cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key
        
        Returns:
            True if key was deleted
        """
        await self.connect()
        
        try:
            result = await self.redis.delete(self._make_key(key))
            return result > 0
        except Exception as e:
            print(f"Cache delete error: {e}")
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching pattern.
        
        Args:
            pattern: Glob pattern (e.g., "user:*")
        
        Returns:
            Number of keys deleted
        """
        await self.connect()
        
        try:
            keys = []
            async for key in self.redis.scan_iter(match=self._make_key(pattern)):
                keys.append(key)
            
            if keys:
                return await self.redis.delete(*keys)
            return 0
        except Exception as e:
            print(f"Cache delete_pattern error: {e}")
            return 0
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: int = TTL_MEDIUM
    ) -> Any:
        """
        Get value from cache or compute and cache it.
        
        Args:
            key: Cache key
            factory: Function to compute value if not cached
            ttl: Time to live in seconds
        
        Returns:
            Cached or computed value
        """
        value = await self.get(key)
        if value is not None:
            return value
        
        # Compute value
        value = factory()
        if hasattr(value, '__await__'):
            value = await value
        
        # Cache it
        await self.set(key, value, ttl)
        return value
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """
        Increment counter.
        
        Args:
            key: Cache key
            amount: Amount to increment
        
        Returns:
            New value
        """
        await self.connect()
        
        try:
            return await self.redis.incrby(self._make_key(key), amount)
        except Exception as e:
            print(f"Cache increment error: {e}")
            return 0
    
    async def rate_limit_check(
        self,
        key: str,
        limit: int,
        window_seconds: int
    ) -> tuple:
        """
        Check rate limit.
        
        Args:
            key: Rate limit key (e.g., "rate:user:123")
            limit: Maximum requests allowed
            window_seconds: Time window in seconds
        
        Returns:
            (allowed: bool, remaining: int, reset_at: int)
        """
        await self.connect()
        
        full_key = self._make_key(key)
        
        try:
            current = await self.redis.get(full_key)
            
            if current is None:
                # First request in window
                await self.redis.setex(full_key, window_seconds, 1)
                return True, limit - 1, window_seconds
            
            current = int(current)
            
            if current >= limit:
                # Rate limited
                ttl = await self.redis.ttl(full_key)
                return False, 0, ttl
            
            # Increment counter
            await self.redis.incr(full_key)
            return True, limit - current - 1, await self.redis.ttl(full_key)
            
        except Exception as e:
            print(f"Rate limit check error: {e}")
            return True, limit, window_seconds  # Fail open
    
    async def acquire_lock(
        self,
        lock_name: str,
        timeout: int = 30
    ) -> Optional[str]:
        """
        Acquire distributed lock.
        
        Args:
            lock_name: Name of lock
            timeout: Lock timeout in seconds
        
        Returns:
            Lock token if acquired, None otherwise
        """
        await self.connect()
        
        import uuid
        token = str(uuid.uuid4())
        key = self._make_key(f"lock:{lock_name}")
        
        try:
            acquired = await self.redis.set(
                key,
                token,
                nx=True,  # Only set if doesn't exist
                ex=timeout,
            )
            return token if acquired else None
        except Exception as e:
            print(f"Lock acquire error: {e}")
            return None
    
    async def release_lock(self, lock_name: str, token: str) -> bool:
        """
        Release distributed lock.
        
        Args:
            lock_name: Name of lock
            token: Lock token from acquire_lock
        
        Returns:
            True if lock was released
        """
        await self.connect()
        
        key = self._make_key(f"lock:{lock_name}")
        
        try:
            # Only release if we own the lock
            current = await self.redis.get(key)
            if current == token:
                await self.redis.delete(key)
                return True
            return False
        except Exception as e:
            print(f"Lock release error: {e}")
            return False
    
    # Convenience methods for common cache patterns
    
    async def cache_user_quota(self, user_id: str, quota: dict) -> bool:
        """Cache user quota information."""
        return await self.set(f"quota:{user_id}", quota, self.TTL_SHORT)
    
    async def get_user_quota(self, user_id: str) -> Optional[dict]:
        """Get cached user quota."""
        return await self.get(f"quota:{user_id}")
    
    async def cache_model_metadata(self, model_id: str, metadata: dict) -> bool:
        """Cache model metadata."""
        return await self.set(f"model:{model_id}", metadata, self.TTL_MEDIUM)
    
    async def get_model_metadata(self, model_id: str) -> Optional[dict]:
        """Get cached model metadata."""
        return await self.get(f"model:{model_id}")
    
    async def cache_job_status(self, job_id: str, status: dict) -> bool:
        """Cache job status (short TTL for real-time updates)."""
        return await self.set(f"job:{job_id}", status, self.TTL_SHORT)
    
    async def get_job_status(self, job_id: str) -> Optional[dict]:
        """Get cached job status."""
        return await self.get(f"job:{job_id}")
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for user."""
        count = 0
        count += await self.delete_pattern(f"quota:{user_id}")
        count += await self.delete_pattern(f"user:{user_id}:*")
        return count


# Singleton instance
_cache_service: Optional[CacheService] = None


def get_cache_service() -> CacheService:
    """Get or create cache service singleton."""
    global _cache_service
    if _cache_service is None:
        _cache_service = CacheService()
    return _cache_service
