"""Application-owned Anki write boundary.

Agents never receive this capability. The only public write operation accepts
a ``PipelineRun`` and checks its explicit authorization state before invoking
the external AnkiConnect adapter.
"""

from __future__ import annotations

from collections.abc import Callable

from .errors import AnkiResponseError, PipelineError
from .tools import push_cards_batch
from .workflow import FailureRecord, PipelineRun


def _failure_record(error: PipelineError) -> FailureRecord:
    """Convert a typed application error into persisted workflow metadata."""
    return FailureRecord(**error.as_dict())


def write_approved_run(
    run: PipelineRun,
    write_batch: Callable[[str], str] = push_cards_batch,
) -> str:
    """Write an explicitly approved run to Anki through this boundary.

    Writes are deliberately not retried here. Until Commit 4 adds idempotency
    and partial-write recovery, an automatic retry could duplicate cards that
    were successfully created before a connection failure.
    """
    run.begin_write()
    payload = run.cards.model_dump_json()

    try:
        result = write_batch(payload)
    except PipelineError as exc:
        run.write_failed(_failure_record(exc))
        raise
    except Exception as exc:
        error = AnkiResponseError(f"Unexpected Anki adapter failure: {exc}")
        run.write_failed(_failure_record(error))
        raise error from exc

    run.write_succeeded()
    return result
