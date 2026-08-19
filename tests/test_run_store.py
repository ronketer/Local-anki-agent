"""Tests for durable local pipeline-run persistence."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.anki_pipeline.models import Flashcard, FlashcardList
from src.anki_pipeline.run_store import RunStore
from src.anki_pipeline.workflow import PipelineRun, WriteItemStatus


def approved_run() -> PipelineRun:
    run = PipelineRun(block_id="20260101000000-test")
    run.content_ready()
    run.draft_ready(
        FlashcardList(
            cards=[
                Flashcard(front="Q1?", back="A1"),
                Flashcard(front="Q2?", back="A2"),
            ]
        )
    )
    run.reviewer_approved()
    run.human_approved()
    return run


def test_save_and_load_round_trip_preserves_approved_manifest(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run_state")
    run = approved_run()

    path = store.save(run)
    loaded = store.load(run.run_id)

    assert path == tmp_path / "run_state" / f"{run.run_id}.json"
    assert loaded == run
    assert [item.status for item in loaded.write_items] == [
        WriteItemStatus.PENDING,
        WriteItemStatus.PENDING,
    ]


def test_save_atomically_replaces_previous_snapshot(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run_state")
    run = approved_run()

    first_path = store.save(run)
    first_payload = first_path.read_text(encoding="utf-8")

    run.begin_write()
    second_path = store.save(run)

    assert second_path == first_path
    assert second_path.read_text(encoding="utf-8") != first_payload
    assert store.load(run.run_id).write_status == run.write_status
    assert list(second_path.parent.glob("*.tmp")) == []
    assert list(second_path.parent.glob(".*.tmp")) == []


@pytest.mark.parametrize("run_id", ["", "../escape", "nested/run", ".", ".."])
def test_rejects_path_like_run_ids(tmp_path: Path, run_id: str) -> None:
    store = RunStore(tmp_path / "run_state")
    run = approved_run()
    run.run_id = run_id

    with pytest.raises(ValueError):
        store.save(run)


def test_load_rejects_corrupt_or_invalid_state(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run_state")
    root = tmp_path / "run_state"
    root.mkdir()

    corrupt_id = "corrupt-run"
    (root / f"{corrupt_id}.json").write_text('{"block_id": 42, "stage": "unknown"}')

    with pytest.raises(ValidationError):
        store.load(corrupt_id)


def test_load_missing_run_raises_file_not_found(tmp_path: Path) -> None:
    store = RunStore(tmp_path / "run_state")

    with pytest.raises(FileNotFoundError):
        store.load("missing-run")
