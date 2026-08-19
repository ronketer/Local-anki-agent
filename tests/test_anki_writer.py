"""Tests for the application-owned Anki write boundary."""

import pytest

from src.anki_pipeline.anki_writer import AnkiWriteError, write_approved_run
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


def test_adapter_exception_marks_run_failed() -> None:
    def failing_writer(payload: str) -> str:
        raise ConnectionError("Anki unavailable")

    run = approved_run()

    with pytest.raises(AnkiWriteError, match="Anki unavailable"):
        write_approved_run(run, write_batch=failing_writer)

    assert run.stage == WorkflowStage.FAILED
    assert run.write_status == WriteStatus.FAILED
    assert run.failure == "Anki unavailable"


def test_legacy_failure_result_marks_run_failed() -> None:
    run = approved_run()

    with pytest.raises(AnkiWriteError, match="Anki not running"):
        write_approved_run(
            run,
            write_batch=lambda payload: "Error: Anki not running",
        )

    assert run.stage == WorkflowStage.FAILED
    assert run.write_status == WriteStatus.FAILED
