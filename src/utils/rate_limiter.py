#!/usr/bin/env python3
"""
Rate limiting middleware for Performance Monitor System API.
Prevents abuse and ensures fair resource allocation.
"""

from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from collections import defaultdict
import asyncio
from fastapi import HTTPException, Request
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Token bucket rate limiter.
    
    Allows burst requests up to bucket size, then enforces rate limit.
    """
    
    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int = 10,
        cleanup_interval: int = 300
    ):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_minute: Maximum requests per minute per client
            burst_size: Maximum burst size allowed
            cleanup_interval: Seconds between cleanup of old entries
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size
        self.cleanup_interval = cleanup_interval
        
        # client_id -> (tokens, last_update)
        self._buckets: Dict[str, Tuple[float, datetime]] = {}
        self._lock = asyncio.Lock()
        self._last_cleanup = datetime.utcnow()
        
        # Refill rate: tokens per second
        self._refill_rate = requests_per_minute / 60.0
    
    def _get_client_id(self, request: Request) -> str:
        """Extract client identifier from request."""
        # Use X-Forwarded-For if behind proxy, otherwise use client IP
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"
    
    async def _cleanup_old_entries(self) -> None:
        """Remove stale bucket entries."""
        now = datetime.utcnow()
        if (now - self._last_cleanup).total_seconds() < self.cleanup_interval:
            return
        
        cutoff = now - timedelta(minutes=5)
        to_remove = [
            client_id for client_id, (_, last_update) in self._buckets.items()
            if last_update < cutoff
        ]
        
        for client_id in to_remove:
            del self._buckets[client_id]
        
        self._last_cleanup = now
        if to_remove:
            logger.debug(f"Cleaned up {len(to_remove)} stale rate limit entries")
    
    async def check_rate_limit(self, request: Request) -> bool:
        """
        Check if request is within rate limit.
        
        Args:
            request: FastAPI request object
            
        Returns:
            True if allowed, raises HTTPException if rate limited
        """
        async with self._lock:
            await self._cleanup_old_entries()
            
            client_id = self._get_client_id(request)
            now = datetime.utcnow()
            
            if client_id not in self._buckets:
                # New client: full bucket
                self._buckets[client_id] = (self.burst_size - 1, now)
                return True
            
            tokens, last_update = self._buckets[client_id]
            
            # Refill tokens based on time elapsed
            elapsed = (now - last_update).total_seconds()
            tokens = min(self.burst_size, tokens + elapsed * self._refill_rate)
            
            if tokens >= 1:
                # Allow request, consume token
                self._buckets[client_id] = (tokens - 1, now)
                return True
            else:
                # Rate limited
                retry_after = int((1 - tokens) / self._refill_rate) + 1
                logger.warning(f"Rate limit exceeded for {client_id}")
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after_seconds": retry_after,
                        "limit": f"{self.requests_per_minute} requests/minute"
                    },
                    headers={"Retry-After": str(retry_after)}
                )
    
    @property
    def stats(self) -> Dict:
        """Get rate limiter statistics."""
        return {
            "active_clients": len(self._buckets),
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size
        }


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""
    
    def __init__(self, app, rate_limiter: RateLimiter, exclude_paths: list = None):
        super().__init__(app)
        self.rate_limiter = rate_limiter
        self.exclude_paths = exclude_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
    
    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for excluded paths
        if any(request.url.path.startswith(path) for path in self.exclude_paths):
            return await call_next(request)
        
        await self.rate_limiter.check_rate_limit(request)
        return await call_next(request)


# Default rate limiter instance
default_rate_limiter = RateLimiter(
    requests_per_minute=120,  # 2 req/sec average
    burst_size=20             # Allow bursts of 20 requests
)
