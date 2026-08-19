"""Typed application and integration failures.

External adapters raise these exceptions instead of returning error strings.
The workflow can therefore classify failures without parsing human-readable
messages.
"""

from __future__ import annotations


class PipelineError(RuntimeError):
    """Base class for expected pipeline failures."""

    retryable: bool = False
    code: str = "pipeline_error"
    service: str | None = None

    def __init__(self, message: str) -> None:
        super().__init__(message)

    def as_dict(self) -> dict[str, object]:
        """Return stable metadata suitable for logs or persisted run state."""
        return {
            "code": self.code,
            "message": str(self),
            "retryable": self.retryable,
            "service": self.service,
        }


class ConfigurationError(PipelineError):
    """Invalid or missing application configuration."""

    code = "configuration_error"


class PayloadValidationError(PipelineError):
    """Application payload is malformed and should not be retried unchanged."""

    code = "payload_validation_error"


class IntegrationError(PipelineError):
    """Base class for expected external-service failures."""

    code = "integration_error"


class TransientIntegrationError(IntegrationError):
    """Temporary integration failure that may be safe to retry."""

    retryable = True
    code = "transient_integration_error"


class PermanentIntegrationError(IntegrationError):
    """Integration failure that should fail fast without retry."""

    code = "permanent_integration_error"


class SiyuanUnavailableError(TransientIntegrationError):
    """Siyuan could not be reached or returned a temporary server failure."""

    code = "siyuan_unavailable"
    service = "siyuan"


class SiyuanResponseError(PermanentIntegrationError):
    """Siyuan rejected the request or returned an invalid response."""

    code = "siyuan_response_error"
    service = "siyuan"


class AnkiUnavailableError(TransientIntegrationError):
    """AnkiConnect could not be reached or returned a temporary server failure."""

    code = "anki_unavailable"
    service = "anki"


class AnkiResponseError(PermanentIntegrationError):
    """AnkiConnect rejected the request or returned an invalid response."""

    code = "anki_response_error"
    service = "anki"
