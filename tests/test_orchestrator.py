"""Tests for deriving trusted workflow state from agent messages."""

from dataclasses import dataclass

from src.anki_pipeline.orchestrator import replay_workflow
from src.anki_pipeline.workflow import HumanDecision, WorkflowStage


@dataclass
class Message:
    source: str
    content: str


CARDS = '{"cards": [{"front": "Q?", "back": "A"}]}'


def test_exact_human_approval_authorizes_validated_cards() -> None:
    run = replay_workflow(
        "20260101000000-test",
        [
            Message("Card_Writer", CARDS),
            Message("Card_Reviewer", "APPROVED"),
            Message("Admin", "APPROVE"),
        ],
    )

    assert run.human_decision == HumanDecision.APPROVED
    assert run.can_write is True


def test_approval_substring_cannot_authorize_write() -> None:
    run = replay_workflow(
        "20260101000000-test",
        [
            Message("Card_Writer", CARDS),
            Message("Card_Reviewer", "APPROVED"),
            Message("Admin", "NOT APPROVED"),
        ],
    )

    assert run.human_decision == HumanDecision.PENDING
    assert run.can_write is False


def test_reviewer_approval_substring_does_not_reach_human_gate() -> None:
    run = replay_workflow(
        "20260101000000-test",
        [
            Message("Card_Writer", CARDS),
            Message("Card_Reviewer", "These are APPROVED"),
            Message("Admin", "APPROVE"),
        ],
    )

    assert run.stage == WorkflowStage.REVIEWING
    assert run.can_write is False


def test_invalid_card_json_cannot_be_approved() -> None:
    run = replay_workflow(
        "20260101000000-test",
        [
            Message("Card_Writer", '{"cards": [{"front": "missing back"}]}'),
            Message("Card_Reviewer", "APPROVED"),
            Message("Admin", "APPROVE"),
        ],
    )

    assert run.stage == WorkflowStage.GENERATING
    assert run.can_write is False


def test_rejection_revision_then_approval_uses_latest_draft() -> None:
    run = replay_workflow(
        "20260101000000-test",
        [
            Message("Card_Writer", '{"cards": [{"front": "Old?", "back": "Old"}]}'),
            Message("Card_Reviewer", "REJECTED:\nToo broad"),
            Message("Card_Writer", '{"cards": [{"front": "New?", "back": "New"}]}'),
            Message("Card_Reviewer", "APPROVED"),
            Message("Admin", "APPROVE"),
        ],
    )

    assert run.cards.cards[0].front == "New?"
    assert run.cards.cards[0].back == "New"
    assert run.can_write is True
