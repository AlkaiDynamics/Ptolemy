"""
Caching Module for Multi-Model Processor

This module provides caching functionality for the Multi-Model Processor
to avoid redundant API calls and improve performance.
"""

import time
import json
import hashlib
from typing import Dict, Any, Optional, Union, Callable
from loguru import logger
import functools


class ResponseCache:
    """
    Cache for model responses.
    
    This class provides a simple in-memory cache for model responses
    with support for TTL and capacity limits.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the response cache.
        
        Args:
            max_size: Maximum number of items to store in the cache
            default_ttl: Default TTL (time to live) in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.hits = 0
        self.misses = 0
        self.lru = []  # LRU tracking list
        logger.info(f"Initialized ResponseCache with max_size={max_size}, default_ttl={default_ttl}s")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get a value from the cache.
        
        Args:
            key: The cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        if key not in self.cache:
            self.misses += 1
            return None
            
        entry = self.cache[key]
        
        # Check if entry is expired
        if entry["expires_at"] < time.time():
            # Remove expired entry
            del self.cache[key]
            self.misses += 1
            return None
            
        # Update access timestamp and count
        entry["last_accessed"] = time.time()
        entry["access_count"] += 1
        
        # Update LRU tracking
        if key in self.lru:
            self.lru.remove(key)
        self.lru.append(key)
        
        self.hits += 1
        logger.debug(f"Cache hit for key: {key[:10]}...")
        return entry["value"]
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """
        Set a value in the cache.
        
        Args:
            key: The cache key
            value: The value to cache
            ttl: TTL in seconds, or None to use the default
        """
        # Check if we need to make room in the cache
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_one()
            
        # Calculate expiration time
        ttl = ttl if ttl is not None else self.default_ttl
        expires_at = time.time() + ttl
        
        # Store the value
        self.cache[key] = {
            "value": value,
            "created_at": time.time(),
            "last_accessed": time.time(),
            "expires_at": expires_at,
            "access_count": 0,
            "ttl": ttl
        }
        
        # Update LRU tracking
        if key in self.lru:
            self.lru.remove(key)
        self.lru.append(key)
        
        logger.debug(f"Cached value for key: {key[:10]}... (TTL: {ttl}s)")
    
    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.
        
        Args:
            key: The cache key
            
        Returns:
            True if the key was found and invalidated, False otherwise
        """
        if key in self.cache:
            del self.cache[key]
            if key in self.lru:
                self.lru.remove(key)
            logger.debug(f"Invalidated cache entry for key: {key[:10]}...")
            return True
        return False
    
    def clear(self) -> None:
        """Clear all entries from the cache."""
        self.cache.clear()
        self.lru.clear()
        logger.info("Cache cleared")
    
    def _evict_one(self) -> None:
        """Evict one item from the cache using LRU policy."""
        if not self.cache:
            return
            
        # Find the least recently used item
        if self.lru:
            # Use the LRU ordering to find the least recently used key
            lru_key = self.lru[0]  # First item is least recently used
            # Remove from LRU tracking
            self.lru.remove(lru_key)
        else:
            # Fallback if LRU list is empty
            lru_key = min(self.cache.keys(), key=lambda k: self.cache[k]["last_accessed"])
        
        # Remove it
        logger.debug(f"Evicted least recently used cache entry: {lru_key[:10]}...")
        del self.cache[lru_key]
    
    async def _evict_expired_items(self):
        """Remove expired items from the cache."""
        now = time.time()
        
        # Find expired entries
        expired_keys = []
        for key, entry in self.cache.items():
            if now - entry["created_at"] > entry["ttl"]:
                expired_keys.append(key)
        
        # Remove expired entries
        for key in expired_keys:
            logger.debug(f"Evicting expired cache entry: {key[:10]}...")
            del self.cache[key]
            if key in self.lru:
                self.lru.remove(key)
                
        return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        # Count expired items
        current_time = time.time()
        expired = sum(1 for entry in self.cache.values() if entry["expires_at"] < current_time)
        
        # Calculate hit rate
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "expired_count": expired,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate
        }


def generate_cache_key(prompt: str, model: str, parameters: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a cache key for a prompt and model.
    
    Args:
        prompt: The prompt text
        model: The model identifier
        parameters: Optional parameters that affect the response
        
    Returns:
        Cache key string
    """
    parameters = parameters or {}
    
    # Filter parameters that affect the response
    relevant_params = {}
    for param in ["temperature", "top_p", "max_tokens", "stop", "presence_penalty", "frequency_penalty"]:
        if param in parameters:
            relevant_params[param] = parameters[param]
    
    # Create a string representation of the key components
    key_data = {
        "prompt": prompt,
        "model": model,
        "parameters": relevant_params
    }
    
    # Generate a hash of the key data
    key_str = json.dumps(key_data, sort_keys=True)
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def with_cache(cache: Optional[Union[ResponseCache, Dict]] = None, ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        cache: Optional ResponseCache instance, creates an in-memory one if not provided
        ttl: Optional time-to-live for cache entries in seconds
        
    Returns:
        Decorated function
    """
    # Create local stats counters
    _hits = 0
    _misses = 0
    
    # Create or use cache
    _local_cache = {}
    _response_cache = None
    
    if cache is None:
        # Use simple in-memory cache
        _cache = _local_cache
    elif isinstance(cache, ResponseCache):
        # Use ResponseCache instance
        _response_cache = cache
        _cache = None  # Will use ResponseCache methods
    else:
        # Assume dict-like object
        _cache = cache
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            nonlocal _hits, _misses
            
            # Extract self for method calls
            if len(args) > 0 and hasattr(args[0], '__class__'):
                instance = args[0]
                # Get prompt and parameters from args
                if len(args) > 1:
                    prompt = args[1]
                    parameters = kwargs.get('parameters', {}) if len(args) <= 2 else args[2]
                else:
                    # Can't determine prompt, skip caching
                    logger.debug("No prompt found, skipping cache")
                    return await func(*args, **kwargs)
            else:
                # Not a method call, use all args
                prompt = args[0] if args else kwargs.get('prompt')
                parameters = kwargs.get('parameters', {}) if len(args) <= 1 else args[1]
                if not prompt:
                    # Can't determine prompt, skip caching
                    logger.debug("No prompt found (non-method call), skipping cache")
                    return await func(*args, **kwargs)
            
            # Skip cache if explicitly requested
            if isinstance(parameters, dict) and parameters.get('skip_cache', False):
                logger.debug("Skip cache requested, bypassing cache")
                return await func(*args, **kwargs)
            
            # Generate cache key
            if _response_cache:
                # Use ResponseCache's key generation
                model_id = getattr(instance, 'model_id', 'unknown') if instance else 'unknown'
                key = _response_cache.generate_key(prompt, model_id, parameters)
                
                # Try to get from cache
                cached_result = await _response_cache.get(key)
                if cached_result:
                    _hits += 1
                    logger.debug(f"Cache hit using ResponseCache for key: {key}")
                    return cached_result
                
                # Cache miss, call original function
                _misses += 1
                logger.debug(f"Cache miss using ResponseCache for key: {key}")
                result = await func(*args, **kwargs)
                
                # Store in cache
                await _response_cache.set(key, result, ttl)
                return result
            else:
                # Use simple dict cache with string key
                key = f"{prompt}:{str(parameters)}"
                
                # Check if in cache
                if key in _cache:
                    _hits += 1
                    logger.debug(f"Cache hit using dict cache for key: {key}")
                    return _cache[key]
                
                # Cache miss, call original function
                _misses += 1
                logger.debug(f"Cache miss using dict cache for key: {key}")
                result = await func(*args, **kwargs)
                
                # Store in cache
                _cache[key] = result
                return result
        
        # Add stats properties and methods
        def get_cache_stats():
            if _response_cache:
                rc_stats = _response_cache.get_stats()
                # Combine with our local counters
                return {
                    "hits": _hits, 
                    "misses": _misses,
                    "size": rc_stats.get("size", 0)
                }
            else:
                return {"hits": _hits, "misses": _misses, "size": len(_cache)}
        
        # Add property for cache stats for test compatibility
        wrapper.cache_stats = property(lambda _: get_cache_stats())
        
        # Add property for cache hits/misses for direct access
        wrapper.cache_hits = property(lambda _: _hits)
        wrapper.cache_misses = property(lambda _: _misses)
        
        # Add reference to cache
        wrapper.cache = _cache if not _response_cache else _response_cache
        
        # Add method to clear stats
        def clear_stats():
            nonlocal _hits, _misses
            _hits = 0
            _misses = 0
        
        wrapper.clear_stats = clear_stats
        
        return wrapper
    
    return decorator
