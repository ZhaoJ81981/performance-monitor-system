#!/usr/bin/env python3
"""
Simple in-memory caching layer for Performance Monitor System.
Reduces redundant database queries and improves response time.
"""

from functools import wraps
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Optional, Tuple
import hashlib
import json
import logging
from threading import Lock

logger = logging.getLogger(__name__)


class TTLCache:
    """Thread-safe TTL cache with size limit."""
    
    def __init__(self, default_ttl: int = 60, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            default_ttl: Default TTL in seconds
            max_size: Maximum number of items in cache
        """
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._lock = Lock()
        self.default_ttl = default_ttl
        self.max_size = max_size
        self._hits = 0
        self._misses = 0
    
    def _make_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        with self._lock:
            if key in self._cache:
                value, expiry = self._cache[key]
                if datetime.utcnow() < expiry:
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with TTL."""
        ttl = ttl or self.default_ttl
        expiry = datetime.utcnow() + timedelta(seconds=ttl)
        
        with self._lock:
            # Evict oldest entries if at capacity
            if len(self._cache) >= self.max_size:
                self._evict_oldest()
            self._cache[key] = (value, expiry)
    
    def _evict_oldest(self) -> None:
        """Remove oldest 10% of entries."""
        if not self._cache:
            return
        
        items = sorted(self._cache.items(), key=lambda x: x[1][1])
        to_remove = max(1, len(items) // 10)
        for key, _ in items[:to_remove]:
            del self._cache[key]
    
    def invalidate(self, key: str) -> None:
        """Remove specific key from cache."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear entire cache."""
        with self._lock:
            self._cache.clear()
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{hit_rate:.1f}%",
            "size": len(self._cache),
            "max_size": self.max_size
        }


# Global cache instances
metrics_cache = TTLCache(default_ttl=30, max_size=500)  # 30s TTL for metrics
prediction_cache = TTLCache(default_ttl=300, max_size=200)  # 5min TTL for predictions
alert_cache = TTLCache(default_ttl=60, max_size=100)  # 1min TTL for alerts


def cached(cache: TTLCache, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache: TTLCache instance to use
        ttl: Optional TTL override in seconds
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Skip caching if explicitly disabled
            skip_cache = kwargs.pop('skip_cache', False)
            if skip_cache:
                return await func(*args, **kwargs)
            
            key = cache._make_key(func.__name__, *args, **kwargs)
            result = cache.get(key)
            
            if result is not None:
                logger.debug(f"Cache hit for {func.__name__}")
                return result
            
            result = await func(*args, **kwargs)
            cache.set(key, result, ttl)
            return result
        
        return wrapper
    return decorator
