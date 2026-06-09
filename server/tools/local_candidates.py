"""Export query-local generated tools as absorber candidates."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from server.tools.local_pool import LocalToolPool
from server.tools.spec import ToolSpec, ToolStatus


class LocalCandidateExporter:
    """Persist query-local generated tools without promoting them to ACTIVE."""

    def __init__(
        self,
        registry: Any,
        telemetry: Any = None,
        candidate_dir: str = "server/tools/candidates",
    ) -> None:
        self._registry = registry
        self._telemetry = telemetry
        self._candidate_dir = Path(candidate_dir)

    def export(self, pool: LocalToolPool, *, task_id: str, batch_id: str = "") -> list[ToolSpec]:
        self._candidate_dir.mkdir(parents=True, exist_ok=True)
        exported: list[ToolSpec] = []

        for spec in pool.list_specs():
            if self._registry.get_tool(spec.name) is not None:
                continue

            source_file = Path(spec.metadata["source_file"])
            candidate_file = self._candidate_dir / f"{spec.name}.py"
            shutil.copyfile(source_file, candidate_file)

            candidate = replace(
                spec,
                provider="query_local_candidate",
                status=ToolStatus.CANDIDATE,
                metadata={
                    **spec.metadata,
                    "absorber_candidate": True,
                    "source": "local_candidate_export",
                    "task_id": task_id,
                    "batch_id": batch_id,
                    "source_file": str(candidate_file),
                },
            )
            self._registry.register(candidate)
            exported.append(candidate)

            if self._telemetry is not None:
                self._telemetry.record_tool_registered(
                    candidate.name,
                    candidate.version,
                    metadata={
                        "source": "local_candidate_export",
                        "task_id": task_id,
                        "batch_id": batch_id,
                    },
                )

        return exported
