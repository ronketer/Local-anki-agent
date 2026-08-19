"""Tests for typed pipeline failures."""

from anki_pipeline.errors import (
    AnkiResponseError,
    AnkiUnavailableError,
    ConfigurationError,
    PayloadValidationError,
    SiyuanResponseError,
    SiyuanUnavailableError,
)


def test_transient_errors_are_retryable() -> None:
    assert SiyuanUnavailableError("down").retryable is True
    assert AnkiUnavailableError("down").retryable is True


def test_permanent_errors_fail_fast() -> None:
    assert SiyuanResponseError("bad response").retryable is False
    assert AnkiResponseError("bad response").retryable is False
    assert PayloadValidationError("bad payload").retryable is False
    assert ConfigurationError("missing setting").retryable is False


def test_error_metadata_is_stable_for_logging() -> None:
    error = AnkiUnavailableError("AnkiConnect timed out")

    assert error.as_dict() == {
        "code": "anki_unavailable",
        "message": "AnkiConnect timed out",
        "retryable": True,
        "service": "anki",
    }
