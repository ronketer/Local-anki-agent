"""Tests for the application-owned idempotent Anki write boundary."""

import pytest

from anki_pipeline.anki_writer import write_approved_run
from anki_pipeline.errors import AnkiResponseError, AnkiUnavailableError
from anki_pipeline.models import Flashcard, FlashcardList
from anki_pipeline.workflow import (
    InvalidWorkflowTransition,
    PipelineRun,
    WorkflowStage,
    WriteItemStatus,
    WriteStatus,
)


def approved_run(card_count: int = 1) -> PipelineRun:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(
        FlashcardList(
            cards=[
                Flashcard(front=f"Q{index + 1}?", back=f"A{index + 1}")
                for index in range(card_count)
            ]
        )
    )
    run.reviewer_approved()
    run.human_approved()
    return run


def snapshot_saver(snapshots: list[PipelineRun]):
    def save(run: PipelineRun) -> object:
        snapshots.append(run.model_copy(deep=True))
        return None

    return save


def test_unapproved_run_never_calls_anki_or_persistence() -> None:
    calls: list[str] = []
    snapshots: list[PipelineRun] = []

    def find_notes(tag: str) -> list[int]:
        calls.append(tag)
        return []

    def add_note(front: str, back: str, tags: list[str]) -> int:
        calls.append(front)
        return 123

    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(FlashcardList(cards=[Flashcard(front="Q?", back="A")]))
    run.reviewer_approved()

    with pytest.raises(InvalidWorkflowTransition):
        write_approved_run(
            run,
            save_run=snapshot_saver(snapshots),
            find_notes=find_notes,
            add_note=add_note,
        )

    assert calls == []
    assert snapshots == []
    assert run.write_status == WriteStatus.NOT_STARTED


def test_approved_run_reconciles_then_creates_and_persists_each_card() -> None:
    find_calls: list[str] = []
    add_calls: list[tuple[str, str, list[str]]] = []
    snapshots: list[PipelineRun] = []
    run = approved_run(card_count=2)

    def find_notes(tag: str) -> list[int]:
        find_calls.append(tag)
        return []

    def add_note(front: str, back: str, tags: list[str]) -> int:
        add_calls.append((front, back, tags))
        return 100 + len(add_calls)

    result = write_approved_run(
        run,
        save_run=snapshot_saver(snapshots),
        find_notes=find_notes,
        add_note=add_note,
    )

    expected_tags = [item.idempotency_key for item in run.write_items]
    assert find_calls == expected_tags
    assert add_calls == [
        ("Q1?", "A1", [expected_tags[0]]),
        ("Q2?", "A2", [expected_tags[1]]),
    ]
    assert [item.anki_note_id for item in run.write_items] == [101, 102]
    assert all(item.status == WriteItemStatus.WRITTEN for item in run.write_items)
    assert run.stage == WorkflowStage.COMPLETED
    assert run.write_status == WriteStatus.SUCCEEDED
    assert run.failure is None
    assert "Card 1: created (101)" in result
    assert "Card 2: created (102)" in result

    assert snapshots[0].stage == WorkflowStage.WRITING
    assert snapshots[0].write_status == WriteStatus.IN_PROGRESS
    assert snapshots[-1].stage == WorkflowStage.COMPLETED
    assert snapshots[-1].write_status == WriteStatus.SUCCEEDED


def test_existing_idempotency_tag_is_reconciled_without_creating_note() -> None:
    add_calls = 0
    snapshots: list[PipelineRun] = []
    run = approved_run()

    def add_note(front: str, back: str, tags: list[str]) -> int:
        nonlocal add_calls
        add_calls += 1
        return 999

    result = write_approved_run(
        run,
        save_run=snapshot_saver(snapshots),
        find_notes=lambda tag: [777],
        add_note=add_note,
    )

    assert add_calls == 0
    assert run.write_items[0].status == WriteItemStatus.WRITTEN
    assert run.write_items[0].anki_note_id == 777
    assert run.write_status == WriteStatus.SUCCEEDED
    assert "already present (777)" in result


def test_partial_failure_preserves_written_failed_and_pending_items() -> None:
    snapshots: list[PipelineRun] = []
    run = approved_run(card_count=3)
    add_calls = 0

    def add_note(front: str, back: str, tags: list[str]) -> int:
        nonlocal add_calls
        add_calls += 1
        if add_calls == 2:
            raise AnkiUnavailableError("request timed out")
        return 501

    with pytest.raises(AnkiUnavailableError, match="request timed out"):
        write_approved_run(
            run,
            save_run=snapshot_saver(snapshots),
            find_notes=lambda tag: [],
            add_note=add_note,
        )

    assert [item.status for item in run.write_items] == [
        WriteItemStatus.WRITTEN,
        WriteItemStatus.FAILED,
        WriteItemStatus.PENDING,
    ]
    assert run.write_items[0].anki_note_id == 501
    assert run.write_items[1].failure is not None
    assert run.write_items[1].failure.retryable is True
    assert run.write_items[2].anki_note_id is None
    assert run.stage == WorkflowStage.FAILED
    assert run.write_status == WriteStatus.PARTIAL
    assert run.failure is not None
    assert run.failure.code == "anki_unavailable"
    assert snapshots[-1].write_status == WriteStatus.PARTIAL


def test_failure_before_any_confirmed_card_is_failed_not_partial() -> None:
    run = approved_run(card_count=2)
    snapshots: list[PipelineRun] = []

    with pytest.raises(AnkiUnavailableError):
        write_approved_run(
            run,
            save_run=snapshot_saver(snapshots),
            find_notes=lambda tag: [],
            add_note=lambda front, back, tags: (_ for _ in ()).throw(
                AnkiUnavailableError("Anki unavailable")
            ),
        )

    assert run.write_status == WriteStatus.FAILED
    assert run.write_items[0].status == WriteItemStatus.FAILED
    assert run.write_items[1].status == WriteItemStatus.PENDING


def test_multiple_notes_for_one_idempotency_tag_fail_safely() -> None:
    run = approved_run()
    add_calls = 0

    def add_note(front: str, back: str, tags: list[str]) -> int:
        nonlocal add_calls
        add_calls += 1
        return 999

    with pytest.raises(AnkiResponseError, match="Multiple Anki notes"):
        write_approved_run(
            run,
            save_run=lambda current: None,
            find_notes=lambda tag: [123, 456],
            add_note=add_note,
        )

    assert add_calls == 0
    assert run.write_status == WriteStatus.FAILED
    assert run.write_items[0].status == WriteItemStatus.FAILED
    assert run.write_items[0].failure is not None
    assert run.write_items[0].failure.retryable is False


def test_unexpected_adapter_exception_is_classified() -> None:
    run = approved_run()

    def broken_find(tag: str) -> list[int]:
        raise RuntimeError("unexpected adapter bug")

    with pytest.raises(AnkiResponseError, match="Unexpected Anki adapter failure"):
        write_approved_run(
            run,
            save_run=lambda current: None,
            find_notes=broken_find,
            add_note=lambda front, back, tags: 123,
        )

    assert run.stage == WorkflowStage.FAILED
    assert run.failure is not None
    assert run.failure.code == "anki_response_error"

def test_resume_skips_confirmed_cards_and_reconciles_ambiguous_failure() -> None:
    run = approved_run(card_count=3)
    snapshots: list[PipelineRun] = []

    run.begin_write()
    run.mark_item_written(0, 501)
    run.mark_item_failed(1, "request timed out")
    run.write_failed("request timed out")

    confirmed_tag = run.write_items[0].idempotency_key
    ambiguous_tag = run.write_items[1].idempotency_key
    missing_tag = run.write_items[2].idempotency_key

    find_calls: list[str] = []
    add_calls: list[tuple[str, str, list[str]]] = []

    def find_notes(tag: str) -> list[int]:
        find_calls.append(tag)
        if tag == ambiguous_tag:
            return [502]
        if tag == missing_tag:
            return []
        raise AssertionError("Already-confirmed card should not be reconciled again")

    def add_note(front: str, back: str, tags: list[str]) -> int:
        add_calls.append((front, back, tags))
        return 503

    result = write_approved_run(
        run,
        save_run=snapshot_saver(snapshots),
        find_notes=find_notes,
        add_note=add_note,
    )

    assert confirmed_tag not in find_calls
    assert find_calls == [ambiguous_tag, missing_tag]
    assert add_calls == [("Q3?", "A3", [missing_tag])]
    assert [item.anki_note_id for item in run.write_items] == [501, 502, 503]
    assert all(item.status == WriteItemStatus.WRITTEN for item in run.write_items)
    assert run.stage == WorkflowStage.COMPLETED
    assert run.write_status == WriteStatus.SUCCEEDED
    assert "Card 1: already confirmed (501)" in result
    assert "Card 2: already present (502)" in result
    assert "Card 3: created (503)" in result


def test_resume_finishes_crashed_run_when_all_cards_were_already_confirmed() -> None:
    run = approved_run(card_count=1)
    snapshots: list[PipelineRun] = []

    run.begin_write()
    run.mark_item_written(0, 777)

    result = write_approved_run(
        run,
        save_run=snapshot_saver(snapshots),
        find_notes=lambda tag: (_ for _ in ()).throw(
            AssertionError("No Anki lookup should be needed")
        ),
        add_note=lambda front, back, tags: (_ for _ in ()).throw(
            AssertionError("No Anki write should be needed")
        ),
    )

    assert run.stage == WorkflowStage.COMPLETED
    assert run.write_status == WriteStatus.SUCCEEDED
    assert "already confirmed (777)" in result

