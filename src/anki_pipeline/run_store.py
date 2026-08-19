"""Durable local persistence for pipeline run state."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from .workflow import PipelineRun

DEFAULT_RUN_STATE_DIR = Path("run_state")


class RunStore:
    """Persist ``PipelineRun`` snapshots as atomically replaced JSON files."""

    def __init__(self, root: str | Path = DEFAULT_RUN_STATE_DIR) -> None:
        self.root = Path(root)

    def _path_for(self, run_id: str) -> Path:
        """Return the state path while rejecting path-like run identifiers."""
        if not run_id or Path(run_id).name != run_id or run_id in {".", ".."}:
            raise ValueError("run_id must be a non-empty file-safe identifier")
        return self.root / f"{run_id}.json"

    def save(self, run: PipelineRun) -> Path:
        """Atomically persist the latest validated run state.

        The temporary file is created in the destination directory so
        ``os.replace`` stays on the same filesystem.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        destination = self._path_for(run.run_id)
        payload = run.model_dump_json(indent=2)

        temp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=f".{run.run_id}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temp_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(temp_path, destination)
        except Exception:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
            raise

        return destination

    def load(self, run_id: str) -> PipelineRun:
        """Load and validate one persisted run snapshot."""
        path = self._path_for(run_id)
        return PipelineRun.model_validate_json(path.read_text(encoding="utf-8"))
