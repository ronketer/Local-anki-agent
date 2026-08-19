"""Tests for deterministic retry policy."""

import pytest

from anki_pipeline.errors import (
    SiyuanResponseError,
    SiyuanUnavailableError,
)
from anki_pipeline.retry import RetryPolicy, retry_call


def test_transient_read_succeeds_after_retries() -> None:
    attempts = 0
    delays: list[float] = []

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise SiyuanUnavailableError("temporarily unavailable")
        return "ok"

    result = retry_call(
        operation,
        policy=RetryPolicy(
            max_attempts=3,
            initial_delay_seconds=0.1,
            backoff_multiplier=2,
            max_delay_seconds=1,
        ),
        sleep=delays.append,
    )

    assert result == "ok"
    assert attempts == 3
    assert delays == [0.1, 0.2]


def test_transient_failure_is_reraised_after_retry_budget() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise SiyuanUnavailableError("still unavailable")

    with pytest.raises(SiyuanUnavailableError, match="still unavailable"):
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=2, initial_delay_seconds=0),
            sleep=lambda delay: None,
        )

    assert attempts == 2


def test_permanent_failure_is_never_retried() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise SiyuanResponseError("invalid block id")

    with pytest.raises(SiyuanResponseError, match="invalid block id"):
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=5, initial_delay_seconds=0),
            sleep=lambda delay: None,
        )

    assert attempts == 1


def test_unexpected_programming_error_is_not_retried() -> None:
    attempts = 0

    def operation() -> str:
        nonlocal attempts
        attempts += 1
        raise ValueError("bug")

    with pytest.raises(ValueError, match="bug"):
        retry_call(
            operation,
            policy=RetryPolicy(max_attempts=5, initial_delay_seconds=0),
            sleep=lambda delay: None,
        )

    assert attempts == 1


@pytest.mark.parametrize(
    ("attempt", "expected"),
    [(1, 0.0), (2, 0.25), (3, 0.5), (4, 1.0), (5, 1.0)],
)
def test_retry_delay_is_bounded(attempt: int, expected: float) -> None:
    policy = RetryPolicy(
        max_attempts=5,
        initial_delay_seconds=0.25,
        backoff_multiplier=2,
        max_delay_seconds=1,
    )

    assert policy.delay_before_attempt(attempt) == expected
