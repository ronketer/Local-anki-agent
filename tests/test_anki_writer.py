"""Tests for the application-owned Anki write boundary."""

import pytest

from src.anki_pipeline.anki_writer import write_approved_run
from src.anki_pipeline.errors import (
    AnkiResponseError,
    AnkiUnavailableError,
)
from src.anki_pipeline.models import Flashcard, FlashcardList
from src.anki_pipeline.workflow import (
    InvalidWorkflowTransition,
    PipelineRun,
    WorkflowStage,
    WriteStatus,
)


def approved_run() -> PipelineRun:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(
        FlashcardList(cards=[Flashcard(front="Q?", back="A")])
    )
    run.reviewer_approved()
    run.human_approved()
    return run


def test_unapproved_run_never_calls_anki_adapter() -> None:
    calls: list[str] = []

    def fake_writer(payload: str) -> str:
        calls.append(payload)
        return "Saved 1 cards"

    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(
        FlashcardList(cards=[Flashcard(front="Q?", back="A")])
    )
    run.reviewer_approved()

    with pytest.raises(InvalidWorkflowTransition):
        write_approved_run(run, write_batch=fake_writer)

    assert calls == []
    assert run.write_status == WriteStatus.NOT_STARTED


def test_approved_run_calls_adapter_and_completes() -> None:
    calls: list[str] = []

    def fake_writer(payload: str) -> str:
        calls.append(payload)
        return "Saved 1 cards:\nCard 1: Card added: 123"

    run = approved_run()

    result = write_approved_run(run, write_batch=fake_writer)

    assert "Card added" in result
    assert len(calls) == 1
    assert '"front":"Q?"' in calls[0]
    assert run.stage == WorkflowStage.COMPLETED
    assert run.write_status == WriteStatus.SUCCEEDED
    assert run.failure is None


def test_typed_transient_failure_is_preserved_in_run_state() -> None:
    calls = 0

    def failing_writer(payload: str) -> str:
        nonlocal calls
        calls += 1
        raise AnkiUnavailableError("Anki unavailable")

    run = approved_run()

    with pytest.raises(AnkiUnavailableError, match="Anki unavailable"):
        write_approved_run(run, write_batch=failing_writer)

    # Writes are not automatically retried until idempotency exists.
    assert calls == 1
    assert run.stage == WorkflowStage.FAILED
    assert run.write_status == WriteStatus.FAILED
    assert run.failure is not None
    assert run.failure.code == "anki_unavailable"
    assert run.failure.retryable is True
    assert run.failure.service == "anki"


def test_permanent_failure_is_preserved_in_run_state() -> None:
    def failing_writer(payload: str) -> str:
        raise AnkiResponseError("duplicate note")

    run = approved_run()

    with pytest.raises(AnkiResponseError, match="duplicate note"):
        write_approved_run(run, write_batch=failing_writer)

    assert run.failure is not None
    assert run.failure.code == "anki_response_error"
    assert run.failure.retryable is False


def test_unexpected_adapter_exception_is_classified() -> None:
    def broken_writer(payload: str) -> str:
        raise RuntimeError("unexpected adapter bug")

    run = approved_run()

    with pytest.raises(AnkiResponseError, match="Unexpected Anki adapter failure"):
        write_approved_run(run, write_batch=broken_writer)

    assert run.stage == WorkflowStage.FAILED
    assert run.failure is not None
    assert run.failure.code == "anki_response_error"
