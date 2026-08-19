"""Application-owned Anki write boundary.

Agents never receive this capability. The only public write operation accepts
a ``PipelineRun`` and checks its explicit authorization state before invoking
the external AnkiConnect adapter.
"""

from __future__ import annotations

from collections.abc import Callable

from .tools import push_cards_batch
from .workflow import PipelineRun


class AnkiWriteError(RuntimeError):
    """Raised when the external Anki write does not complete successfully."""


def _result_indicates_failure(result: str) -> bool:
    """Interpret the legacy string result returned by the Anki adapter."""
    return any(
        line.strip().startswith(("Error:", "Failed:"))
        or ": Error:" in line
        or ": Failed:" in line
        for line in result.splitlines()
    )


def write_approved_run(
    run: PipelineRun,
    write_batch: Callable[[str], str] = push_cards_batch,
) -> str:
    """Write an explicitly approved run to Anki exactly through this boundary.

    ``PipelineRun.begin_write`` is called before the adapter, so an unapproved
    run fails before any external side effect is attempted.
    """
    run.begin_write()
    payload = run.cards.model_dump_json()

    try:
        result = write_batch(payload)
    except Exception as exc:
        run.write_failed(str(exc))
        raise AnkiWriteError(str(exc)) from exc

    if _result_indicates_failure(result):
        run.write_failed(result)
        raise AnkiWriteError(result)

    run.write_succeeded()
    return result
