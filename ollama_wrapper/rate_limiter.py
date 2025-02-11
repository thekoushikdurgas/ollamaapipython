"""Rate limiting implementation for Ollama API"""
import asyncio
import time
from typing import Optional, Dict

class TokenBucket:
    """Token bucket rate limiter implementation"""
    def __init__(self, rate: float, capacity: float):
        """Initialize token bucket
        Args:
            rate (float): Rate of token replenishment per second
            capacity (float): Maximum number of tokens that can be stored
        """
        self.rate = rate
        self.capacity = capacity
        self.current_tokens = capacity
        self.last_update = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> float:
        """Acquire tokens from the bucket
        Args:
            tokens (float): Number of tokens to acquire
        Returns:
            float: Time to wait before acquiring tokens
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self.last_update
            self.current_tokens = min(
                self.capacity,
                self.current_tokens + elapsed * self.rate
            )
            self.last_update = now

            wait_time = (tokens - self.current_tokens) / self.rate if tokens > self.current_tokens else 0
            if wait_time <= 0:
                self.current_tokens -= tokens
                return 0

            return wait_time

class RateLimiter:
    """Rate limiter for Ollama API"""
    def __init__(self):
        self._buckets: Dict[str, TokenBucket] = {}
        self._default_rate = 10  # requests per second
        self._default_capacity = 10

    def get_bucket(self, key: str, rate: Optional[float] = None, capacity: Optional[float] = None) -> TokenBucket:
        """Get or create a token bucket for the given key"""
        if key not in self._buckets:
            self._buckets[key] = TokenBucket(
                rate or self._default_rate,
                capacity or self._default_capacity
            )
        return self._buckets[key]

    async def acquire(self, key: str, tokens: float = 1.0) -> None:
        """Acquire tokens for the given key
        Args:
            key (str): Rate limit key (e.g. endpoint name)
            tokens (float): Number of tokens to acquire
        """
        bucket = self.get_bucket(key)
        wait_time = await bucket.acquire(tokens)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
