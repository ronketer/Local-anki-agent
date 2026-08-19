"""Deterministic workflow state for a single flashcard-generation run.

The LLM agents generate and review content, but they do not own workflow
authorization. This module models the application-owned state machine that
later side-effecting code can consult before writing to Anki.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import FlashcardList


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class WorkflowStage(StrEnum):
    """Lifecycle stages for a pipeline run."""

    FETCHING = "fetching"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    AWAITING_HUMAN = "awaiting_human"
    WRITING = "writing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ReviewDecision(StrEnum):
    """Normalized reviewer decisions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class HumanDecision(StrEnum):
    """Normalized human decisions."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class WriteStatus(StrEnum):
    """State of the external Anki write."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class InvalidWorkflowTransition(RuntimeError):
    """Raised when application code attempts an invalid state transition."""


class FailureRecord(BaseModel):
    """Structured failure metadata persisted with a pipeline run."""

    code: str
    message: str
    retryable: bool = False
    service: str | None = None


class PipelineRun(BaseModel):
    """Explicit, serializable state for one pipeline execution."""

    run_id: str = Field(default_factory=lambda: str(uuid4()))
    block_id: str
    stage: WorkflowStage = WorkflowStage.FETCHING
    cards: FlashcardList = Field(default_factory=lambda: FlashcardList(cards=[]))
    review_decision: ReviewDecision = ReviewDecision.PENDING
    human_decision: HumanDecision = HumanDecision.PENDING
    rejection_count: int = 0
    max_rejections: int = 2
    write_status: WriteStatus = WriteStatus.NOT_STARTED
    failure: FailureRecord | None = None
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    def _require_stage(self, *allowed: WorkflowStage) -> None:
        if self.stage not in allowed:
            expected = ", ".join(stage.value for stage in allowed)
            raise InvalidWorkflowTransition(
                f"Cannot transition from {self.stage.value}; expected one of: {expected}"
            )

    def _touch(self) -> None:
        self.updated_at = _utc_now()

    def content_ready(self) -> None:
        """Move from source retrieval to card generation."""
        self._require_stage(WorkflowStage.FETCHING)
        self.stage = WorkflowStage.GENERATING
        self._touch()

    def draft_ready(self, cards: FlashcardList) -> None:
        """Store the latest validated draft and send it to review."""
        self._require_stage(WorkflowStage.GENERATING)
        self.cards = cards
        self.review_decision = ReviewDecision.PENDING
        self.human_decision = HumanDecision.PENDING
        self.stage = WorkflowStage.REVIEWING
        self._touch()

    def reviewer_approved(self) -> None:
        """Accept the automated review and wait for a human decision."""
        self._require_stage(WorkflowStage.REVIEWING)
        self.review_decision = ReviewDecision.APPROVED
        self.stage = WorkflowStage.AWAITING_HUMAN
        self._touch()

    def reviewer_rejected(self) -> None:
        """Return to generation, or escalate after the configured retry cap."""
        self._require_stage(WorkflowStage.REVIEWING)
        self.review_decision = ReviewDecision.REJECTED
        self.rejection_count += 1
        if self.rejection_count >= self.max_rejections:
            self.stage = WorkflowStage.AWAITING_HUMAN
        else:
            self.stage = WorkflowStage.GENERATING
        self._touch()

    def human_rejected(self) -> None:
        """Return human-rejected cards to the writer for revision."""
        self._require_stage(WorkflowStage.AWAITING_HUMAN)
        self.human_decision = HumanDecision.REJECTED
        self.stage = WorkflowStage.GENERATING
        self._touch()

    def human_approved(self) -> None:
        """Record explicit human authorization without performing a side effect."""
        self._require_stage(WorkflowStage.AWAITING_HUMAN)
        if not self.cards.cards:
            raise InvalidWorkflowTransition("Cannot approve a run with no validated cards")
        self.human_decision = HumanDecision.APPROVED
        self._touch()

    @property
    def can_write(self) -> bool:
        """Whether application code is currently authorized to write to Anki."""
        return (
            self.stage == WorkflowStage.AWAITING_HUMAN
            and self.human_decision == HumanDecision.APPROVED
            and bool(self.cards.cards)
            and self.write_status == WriteStatus.NOT_STARTED
        )

    def begin_write(self) -> None:
        """Enter the side-effecting stage only after explicit human approval."""
        if not self.can_write:
            raise InvalidWorkflowTransition(
                "Anki write requires validated cards and explicit human approval"
            )
        self.stage = WorkflowStage.WRITING
        self.write_status = WriteStatus.IN_PROGRESS
        self._touch()

    def write_succeeded(self) -> None:
        """Mark the external write as complete."""
        self._require_stage(WorkflowStage.WRITING)
        self.write_status = WriteStatus.SUCCEEDED
        self.stage = WorkflowStage.COMPLETED
        self._touch()

    def write_failed(self, failure: FailureRecord | str) -> None:
        """Record an unsuccessful write attempt with stable failure metadata."""
        self._require_stage(WorkflowStage.WRITING)
        self.write_status = WriteStatus.FAILED
        self.failure = (
            failure
            if isinstance(failure, FailureRecord)
            else FailureRecord(code="write_failed", message=failure)
        )
        self.stage = WorkflowStage.FAILED
        self._touch()


def parse_review_decision(content: str) -> ReviewDecision | None:
    """Parse the reviewer's protocol without unsafe substring matching.

    ``APPROVED`` must be the complete trimmed response. ``REJECTED`` may be
    followed by feedback because the reviewer prompt asks it to list fixes.
    """
    normalized = content.strip()
    if normalized.upper() == "APPROVED":
        return ReviewDecision.APPROVED

    first_line = normalized.splitlines()[0].strip().upper() if normalized else ""
    if first_line == "REJECTED" or first_line.startswith("REJECTED:"):
        return ReviewDecision.REJECTED

    return None


def parse_human_decision(content: str) -> HumanDecision | None:
    """Parse explicit terminal human decisions.

    Only the exact word ``APPROVE`` authorizes a side effect. Text such as
    ``NOT APPROVED`` or ``I approve these`` deliberately does not.
    """
    normalized = content.strip().upper()
    if normalized == "APPROVE":
        return HumanDecision.APPROVED
    if normalized == "REJECT":
        return HumanDecision.REJECTED
    return None
