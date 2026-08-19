"""Deterministic retry policy for safe external operations.

The retry helper is deliberately generic, but callers must only use it for
operations that are safe to repeat. Read-only Siyuan fetches are retryable.
Anki writes are intentionally *not* automatically retried until the write
path is idempotent.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .errors import TransientIntegrationError

T = TypeVar("T")


@dataclass(frozen=True)
class RetryPolicy:
    """Bounded exponential-backoff policy."""

    max_attempts: int = 3
    initial_delay_seconds: float = 0.25
    backoff_multiplier: float = 2.0
    max_delay_seconds: float = 2.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.initial_delay_seconds < 0:
            raise ValueError("initial_delay_seconds cannot be negative")
        if self.backoff_multiplier < 1:
            raise ValueError("backoff_multiplier must be at least 1")
        if self.max_delay_seconds < 0:
            raise ValueError("max_delay_seconds cannot be negative")

    def delay_before_attempt(self, attempt: int) -> float:
        """Delay before the given retry attempt, where attempt 2 is first retry."""
        if attempt <= 1:
            return 0.0
        delay = self.initial_delay_seconds * (
            self.backoff_multiplier ** (attempt - 2)
        )
        return min(delay, self.max_delay_seconds)


DEFAULT_READ_RETRY_POLICY = RetryPolicy()


def retry_call(
    operation: Callable[[], T],
    *,
    policy: RetryPolicy = DEFAULT_READ_RETRY_POLICY,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run a safe operation, retrying only typed transient failures."""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except TransientIntegrationError:
            if attempt == policy.max_attempts:
                raise
            sleep(policy.delay_before_attempt(attempt + 1))

    raise AssertionError("retry loop exhausted without returning or raising")
