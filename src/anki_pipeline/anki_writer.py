"""Application-owned idempotent Anki write boundary.

Agents never receive this capability. The public write operation accepts an
explicitly approved ``PipelineRun``, reconciles each card against Anki using an
application-owned tag, and persists progress after every confirmed write.
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import AnkiResponseError, PipelineError
from .tools import add_note as add_note_to_anki, find_note_ids_by_tag
from .workflow import FailureRecord, PipelineRun


PersistRun = Callable[[PipelineRun], object]
FindNotes = Callable[[str], list[int]]
AddNote = Callable[[str, str, list[str]], int]


def _failure_record(error: PipelineError) -> FailureRecord:
    """Convert a typed application error into persisted workflow metadata."""
    return FailureRecord(**error.as_dict())


def write_approved_run(
    run: PipelineRun,
    *,
    save_run: PersistRun,
    find_notes: FindNotes = find_note_ids_by_tag,
    add_note: AddNote = add_note_to_anki,
) -> str:
    """Write an approved run without duplicating already-confirmed cards.

    Each card is first reconciled by its application-owned idempotency tag. A
    missing tag is written once; an existing tag is treated as already written.
    Progress is persisted after entering the write stage, after every confirmed
    card, and after terminal success or failure.
    """
    run.begin_write()
    save_run(run)

    results: list[str] = []

    for item in run.write_items:
        card = run.cards.cards[item.index]

        try:
            note_ids = find_notes(item.idempotency_key)
            if len(note_ids) > 1:
                raise AnkiResponseError(
                    "Multiple Anki notes found for idempotency tag "
                    f"{item.idempotency_key}"
                )

            if note_ids:
                note_id = note_ids[0]
                outcome = "already present"
            else:
                note_id = add_note(
                    card.front,
                    card.back,
                    [item.idempotency_key],
                )
                outcome = "created"
        except PipelineError as exc:
            failure = _failure_record(exc)
            run.mark_item_failed(item.index, failure)
            run.write_failed(failure)
            save_run(run)
            raise
        except Exception as exc:
            error = AnkiResponseError(f"Unexpected Anki adapter failure: {exc}")
            failure = _failure_record(error)
            run.mark_item_failed(item.index, failure)
            run.write_failed(failure)
            save_run(run)
            raise error from exc

        run.mark_item_written(item.index, note_id)
        save_run(run)
        results.append(f"Card {item.index + 1}: {outcome} ({note_id})")

    run.write_succeeded()
    save_run(run)
    return f"Saved {len(results)} cards:\n" + "\n".join(results)
