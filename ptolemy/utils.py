"""
Utility functions for the PTOLEMY project.
"""
import asyncio
import random
from functools import wraps
from loguru import logger

def async_retry(max_retries=3, base_delay=1, backoff_factor=2, jitter=0.1):
    """
    Decorator for async functions to implement retry with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay on each retry
        jitter: Random jitter factor to avoid thundering herd
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    retries += 1
                    if retries > max_retries:
                        logger.error(f"Failed after {max_retries} retries: {str(e)}")
                        raise
                    
                    # Calculate delay with jitter
                    delay = base_delay * (backoff_factor ** (retries - 1))
                    jitter_amount = delay * jitter * random.uniform(-1, 1)
                    delay = max(0.1, delay + jitter_amount)
                    
                    logger.warning(f"Retry {retries}/{max_retries} after {delay:.2f}s: {str(e)}")
                    await asyncio.sleep(delay)
        return wrapper
    return decorator
