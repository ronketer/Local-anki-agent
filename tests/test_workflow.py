"""Tests for deterministic workflow state and approval parsing."""

import pytest

from src.anki_pipeline.models import Flashcard, FlashcardList
from src.anki_pipeline.workflow import (
    HumanDecision,
    InvalidWorkflowTransition,
    PipelineRun,
    ReviewDecision,
    WorkflowStage,
    WriteStatus,
    parse_human_decision,
    parse_review_decision,
)


@pytest.fixture
def cards() -> FlashcardList:
    return FlashcardList(cards=[Flashcard(front="Q?", back="A")])


def test_human_approval_requires_exact_terminal_command() -> None:
    assert parse_human_decision("APPROVE") == HumanDecision.APPROVED
    assert parse_human_decision(" approve \n") == HumanDecision.APPROVED

    assert parse_human_decision("NOT APPROVED") is None
    assert parse_human_decision("I APPROVE") is None
    assert parse_human_decision("APPROVED") is None
    assert parse_human_decision("APPROVE these cards") is None


def test_human_rejection_requires_exact_terminal_command() -> None:
    assert parse_human_decision("REJECT") == HumanDecision.REJECTED
    assert parse_human_decision(" reject ") == HumanDecision.REJECTED
    assert parse_human_decision("REJECTED") is None


def test_review_decision_uses_explicit_protocol() -> None:
    assert parse_review_decision("APPROVED") == ReviewDecision.APPROVED
    assert parse_review_decision("REJECTED") == ReviewDecision.REJECTED
    assert parse_review_decision("REJECTED:\nCard 2 is too broad") == ReviewDecision.REJECTED

    assert parse_review_decision("NOT APPROVED") is None
    assert parse_review_decision("These are APPROVED") is None


def test_happy_path_requires_human_approval_before_write(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")

    run.content_ready()
    run.draft_ready(cards)
    run.reviewer_approved()

    assert run.stage == WorkflowStage.AWAITING_HUMAN
    assert run.can_write is False

    run.human_approved()

    assert run.can_write is True

    run.begin_write()
    assert run.stage == WorkflowStage.WRITING
    assert run.write_status == WriteStatus.IN_PROGRESS

    run.write_succeeded()
    assert run.stage == WorkflowStage.COMPLETED
    assert run.write_status == WriteStatus.SUCCEEDED


def test_write_is_rejected_without_human_approval(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(cards)
    run.reviewer_approved()

    with pytest.raises(InvalidWorkflowTransition):
        run.begin_write()


def test_human_cannot_approve_before_review(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(cards)

    with pytest.raises(InvalidWorkflowTransition):
        run.human_approved()


def test_reviewer_rejection_returns_to_generation(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(cards)

    run.reviewer_rejected()

    assert run.rejection_count == 1
    assert run.stage == WorkflowStage.GENERATING
    assert run.review_decision == ReviewDecision.REJECTED


def test_reviewer_rejection_cap_escalates_to_human(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test", max_rejections=2)
    run.content_ready()
    run.draft_ready(cards)
    run.reviewer_rejected()

    run.draft_ready(cards)
    run.reviewer_rejected()

    assert run.rejection_count == 2
    assert run.stage == WorkflowStage.AWAITING_HUMAN
    assert run.can_write is False


def test_human_rejection_never_authorizes_write(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(cards)
    run.reviewer_approved()
    run.human_rejected()

    assert run.stage == WorkflowStage.GENERATING
    assert run.human_decision == HumanDecision.REJECTED
    assert run.can_write is False


def test_failed_write_is_recorded(cards: FlashcardList) -> None:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(cards)
    run.reviewer_approved()
    run.human_approved()
    run.begin_write()

    run.write_failed("AnkiConnect unavailable")

    assert run.stage == WorkflowStage.FAILED
    assert run.write_status == WriteStatus.FAILED
    assert run.failure is not None
    assert run.failure.code == "write_failed"
    assert run.failure.message == "AnkiConnect unavailable"
    assert run.failure.retryable is False
