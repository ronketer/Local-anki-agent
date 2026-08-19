"""Deterministic workflow state for a single flashcard-generation run.

The LLM agents generate and review content, but they do not own workflow
authorization. This module models the application-owned state machine that
later side-effecting code can consult before writing to Anki.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from hashlib import sha256
from uuid import uuid4

from pydantic import BaseModel, Field

from .models import Flashcard, FlashcardList

IDEMPOTENCY_KEY_PREFIX = "local_anki_agent_id_"
IDEMPOTENCY_DIGEST_LENGTH = 24


def _utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


def _card_idempotency_key(run_id: str, index: int, card: Flashcard) -> str:
    """Return a stable run-scoped key for one approved card.

    The exact approved card payload is included so the persisted manifest can
    later be reconciled with Anki without treating similar cards from another
    run as duplicates.
    """
    material = "\0".join((run_id, str(index), card.front, card.back))
    digest = sha256(material.encode("utf-8")).hexdigest()[:IDEMPOTENCY_DIGEST_LENGTH]
    return f"{IDEMPOTENCY_KEY_PREFIX}{digest}"


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
    """Aggregate state of external Anki writes for a run."""

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    PARTIAL = "partial"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class WriteItemStatus(StrEnum):
    """Durable write state for one approved card."""

    PENDING = "pending"
    WRITTEN = "written"
    FAILED = "failed"


class InvalidWorkflowTransition(RuntimeError):
    """Raised when application code attempts an invalid state transition."""


class FailureRecord(BaseModel):
    """Structured failure metadata persisted with a pipeline run."""

    code: str
    message: str
    retryable: bool = False
    service: str | None = None


class WriteItem(BaseModel):
    """Durable Anki-write metadata for one approved card."""

    index: int
    idempotency_key: str
    status: WriteItemStatus = WriteItemStatus.PENDING
    anki_note_id: int | None = None
    failure: FailureRecord | None = None


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
    write_items: list[WriteItem] = Field(default_factory=list)
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

    def _initialize_write_manifest(self) -> None:
        """Snapshot the exact approved cards into durable write metadata."""
        self.write_items = [
            WriteItem(
                index=index,
                idempotency_key=_card_idempotency_key(self.run_id, index, card),
            )
            for index, card in enumerate(self.cards.cards)
        ]

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
        self.write_status = WriteStatus.NOT_STARTED
        self.write_items = []
        self.failure = None
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
        self.write_items = []
        self.stage = WorkflowStage.GENERATING
        self._touch()

    def human_approved(self) -> None:
        """Record explicit human authorization and snapshot the write manifest."""
        self._require_stage(WorkflowStage.AWAITING_HUMAN)
        if not self.cards.cards:
            raise InvalidWorkflowTransition("Cannot approve a run with no validated cards")
        self.human_decision = HumanDecision.APPROVED
        self._initialize_write_manifest()
        self._touch()

    @property
    def can_write(self) -> bool:
        """Whether application code is currently authorized to write to Anki."""
        manifest_matches_cards = len(self.write_items) == len(self.cards.cards) and all(
            item.index == index for index, item in enumerate(self.write_items)
        )
        return (
            self.stage == WorkflowStage.AWAITING_HUMAN
            and self.human_decision == HumanDecision.APPROVED
            and bool(self.cards.cards)
            and manifest_matches_cards
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

    def mark_item_written(self, index: int, note_id: int) -> None:
        """Record one card as confirmed in Anki."""
        self._require_stage(WorkflowStage.WRITING)
        try:
            item = self.write_items[index]
        except IndexError as exc:
            raise InvalidWorkflowTransition(f"Unknown write item index: {index}") from exc

        if item.index != index:
            raise InvalidWorkflowTransition("Write manifest indices are inconsistent")
        if note_id <= 0:
            raise InvalidWorkflowTransition("Anki note id must be positive")

        item.status = WriteItemStatus.WRITTEN
        item.anki_note_id = note_id
        item.failure = None
        self._touch()

    def mark_item_failed(self, index: int, failure: FailureRecord | str) -> None:
        """Record a failed or ambiguous write attempt for one approved card."""
        self._require_stage(WorkflowStage.WRITING)
        try:
            item = self.write_items[index]
        except IndexError as exc:
            raise InvalidWorkflowTransition(f"Unknown write item index: {index}") from exc

        if item.index != index:
            raise InvalidWorkflowTransition("Write manifest indices are inconsistent")

        item.status = WriteItemStatus.FAILED
        item.failure = (
            failure
            if isinstance(failure, FailureRecord)
            else FailureRecord(code="write_failed", message=failure)
        )
        self._touch()

    def write_succeeded(self) -> None:
        """Complete the run only when every approved card is confirmed written."""
        self._require_stage(WorkflowStage.WRITING)
        if not self.write_items or any(
            item.status != WriteItemStatus.WRITTEN for item in self.write_items
        ):
            raise InvalidWorkflowTransition(
                "Cannot complete Anki write while manifest items remain unconfirmed"
            )

        self.write_status = WriteStatus.SUCCEEDED
        self.failure = None
        self.stage = WorkflowStage.COMPLETED
        self._touch()

    def write_failed(self, failure: FailureRecord | str) -> None:
        """Record an unsuccessful write attempt with stable failure metadata."""
        self._require_stage(WorkflowStage.WRITING)
        self.write_status = (
            WriteStatus.PARTIAL
            if any(item.status == WriteItemStatus.WRITTEN for item in self.write_items)
            else WriteStatus.FAILED
        )
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
