"""ToolEvolutionOrchestrator — compose the tool evolution pipeline.

Composes existing components (ToolRegistry, TelemetryService, ToolDeveloper,
ToolVerifier, ToolAbsorber, ToolRetriever) into a single entry point for the
tool evolution lifecycle:

1. **requests**  — consume ToolRequests from telemetry, generate+verify+register.
2. **absorber**  — find and merge similar tools.
3. **promote**   — promote CANDIDATE tools to ACTIVE based on telemetry stats.
4. **metrics**   — compute system-wide evolution metrics.

All methods accept dependencies via constructor injection for testability.
All errors are logged and surfaced in result dicts — never silently swallowed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from server.tools.spec import RiskLevel, ToolRequest, ToolStatus

logger = logging.getLogger(__name__)

# Risk levels that are safe for auto-generation (L0, L1).
_AUTOGEN_RISK_LEVELS = frozenset({RiskLevel.L0, RiskLevel.L1})


class ToolEvolutionOrchestrator:
    """Top-level orchestrator for the tool evolution pipeline.

    Args:
        registry: ToolRegistry instance (must be connected).
        telemetry: TelemetryService instance (must be connected).
        developer: ToolDeveloper for code generation.
        verifier: ToolVerifier for code validation.
        absorber: ToolAbsorber for tool merging.
        retriever: ToolRetriever for semantic tool search (optional).
    """

    def __init__(
        self,
        registry: Any,
        telemetry: Any,
        developer: Any,
        verifier: Any,
        absorber: Any,
        retriever: Any = None,
    ) -> None:
        self._registry = registry
        self._telemetry = telemetry
        self._developer = developer
        self._verifier = verifier
        self._absorber = absorber
        self._retriever = retriever

    # ------------------------------------------------------------------
    # Core pipeline methods
    # ------------------------------------------------------------------

    async def process_pending_requests(
        self, limit: int = 5, dry_run: bool = False
    ) -> dict[str, Any]:
        """Consume ToolRequests from telemetry: generate -> verify -> register.

        L2+ risk level requests are skipped with a logged reason.
        Requests whose candidate_name already exists in the registry are skipped.

        Returns a structured dict with ``processed``, ``skipped``, ``errors``,
        and ``details`` keys.
        """
        result: dict[str, Any] = {
            "processed": 0,
            "skipped": 0,
            "errors": 0,
            "details": [],
        }

        # Fetch recent tool requests from telemetry.
        requests = self._telemetry.get_tool_requests(limit=limit)
        if not requests:
            logger.info("No pending tool requests found")
            return result

        # Build a set of already-registered tool names for deduplication.
        existing_names = {s.name for s in self._registry.list_tools(status=None)}

        for row in requests:
            detail = self._build_detail_from_row(row)

            # Skip L2+ risk level.
            risk_str = row.get("risk_level", "L0")
            try:
                risk_level = RiskLevel(risk_str)
            except ValueError:
                risk_level = RiskLevel.L0

            if risk_level not in _AUTOGEN_RISK_LEVELS:
                reason = (
                    f"Risk level {risk_level.value} exceeds L0/L1 "
                    "for auto-generation; skipping"
                )
                logger.warning(
                    "Skipping tool request '%s': %s",
                    row.get("candidate_name", "unknown"),
                    reason,
                )
                detail["action"] = "skipped"
                detail["reason"] = reason
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            # Skip already-registered tools.
            candidate_name = row.get("candidate_name", "")
            if candidate_name and candidate_name in existing_names:
                reason = f"Tool '{candidate_name}' already exists in registry"
                logger.info("Skipping tool request: %s", reason)
                detail["action"] = "skipped"
                detail["reason"] = reason
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            # Build a ToolRequest from the DB row.
            tool_request = self._row_to_tool_request(row)

            if dry_run:
                detail["action"] = "dry_run"
                detail["would_generate"] = candidate_name
                result["processed"] += 1
                result["details"].append(detail)
                continue

            # Generate.
            try:
                gen_result = await self._developer.generate(tool_request)
            except Exception as exc:
                reason = f"Generation failed: {exc}"
                logger.error("Tool generation error for '%s': %s", candidate_name, exc)
                detail["action"] = "error"
                detail["reason"] = reason
                result["errors"] += 1
                result["details"].append(detail)
                continue

            if not gen_result.get("success"):
                reason = gen_result.get("error", "unknown generation error")
                logger.warning("Generation failed for '%s': %s", candidate_name, reason)
                detail["action"] = "generation_failed"
                detail["reason"] = reason
                result["errors"] += 1
                result["details"].append(detail)
                continue

            # Verify.
            source_code = gen_result.get("source_code", "")
            try:
                verify_result = self._verifier.verify(candidate_name, source_code)
            except Exception as exc:
                reason = f"Verification error: {exc}"
                logger.error("Verification error for '%s': %s", candidate_name, exc)
                detail["action"] = "error"
                detail["reason"] = reason
                result["errors"] += 1
                result["details"].append(detail)
                continue

            if not verify_result.passed:
                reason = f"Verification failed: {verify_result.errors}"
                logger.warning("Verification failed for '%s': %s", candidate_name, reason)
                detail["action"] = "verify_failed"
                detail["errors"] = verify_result.errors
                result["errors"] += 1
                result["details"].append(detail)
                continue

            # Register as CANDIDATE (not ACTIVE yet — needs promotion).
            from server.tools.spec import ToolSpec, ToolRuntime

            spec = ToolSpec(
                name=candidate_name,
                description=tool_request.candidate_description,
                input_schema=tool_request.candidate_input_schema,
                output_schema=tool_request.candidate_output_schema,
                runtime=ToolRuntime.PYTHON_GENERATED,
                provider="generated",
                entrypoint=gen_result.get("file_path", ""),
                risk_level=risk_level,
                status=ToolStatus.CANDIDATE,
                metadata={
                    "source": "tool_request",
                    "task_id": tool_request.task_id,
                    "session_id": tool_request.session_id,
                    "missing_capability": tool_request.missing_capability,
                },
            )

            try:
                self._registry.register(spec)
            except Exception as exc:
                reason = f"Registration failed: {exc}"
                logger.error("Registration error for '%s': %s", candidate_name, exc)
                detail["action"] = "error"
                detail["reason"] = reason
                result["errors"] += 1
                result["details"].append(detail)
                continue

            # Record creation event in telemetry.
            self._telemetry.record_tool_created(
                candidate_name,
                spec.version,
                metadata={
                    "runtime": "python_generated",
                    "provider": "generated",
                    "source": "tool_request",
                    "risk_level": risk_level.value,
                },
            )

            detail["action"] = "registered"
            detail["status"] = "candidate"
            result["processed"] += 1
            result["details"].append(detail)

            # Track as existing for subsequent iterations in this batch.
            existing_names.add(candidate_name)

        return result

    async def run_absorber(
        self, dry_run: bool = True, max_clusters: int = 5
    ) -> dict[str, Any]:
        """Run the tool absorber to find and merge similar tools.

        Args:
            dry_run: If True, only report what would be merged.
            max_clusters: Maximum number of clusters to process.

        Returns absorber stats with the ``max_clusters`` limit noted.
        """
        try:
            absorber_result = await self._absorber.run(dry_run=dry_run)
        except Exception as exc:
            logger.error("Absorber run failed: %s", exc)
            return {
                "clusters_found": 0,
                "clusters_merged": 0,
                "tools_deprecated": 0,
                "max_clusters": max_clusters,
                "error": str(exc),
                "details": [],
            }

        absorber_result["max_clusters"] = max_clusters
        logger.info(
            "Absorber complete: %d clusters found, %d merged (max_clusters=%d, dry_run=%s)",
            absorber_result.get("clusters_found", 0),
            absorber_result.get("clusters_merged", 0),
            max_clusters,
            dry_run,
        )
        return absorber_result

    def promote_candidates(
        self, min_success_count: int = 3, min_success_rate: float = 0.8
    ) -> dict[str, Any]:
        """Promote CANDIDATE tools to ACTIVE based on telemetry success stats.

        A candidate is promoted when:
        - tool_succeeded count >= ``min_success_count``
        - success rate (succeeded / invoked) >= ``min_success_rate``

        Returns a dict with ``promoted``, ``skipped``, and ``details`` keys.
        """
        result: dict[str, Any] = {
            "promoted": 0,
            "skipped": 0,
            "details": [],
        }

        candidates = self._registry.list_tools(status=ToolStatus.CANDIDATE)
        if not candidates:
            logger.info("No CANDIDATE tools to evaluate for promotion")
            return result

        now = datetime.now(timezone.utc).isoformat()

        for spec in candidates:
            detail: dict[str, Any] = {
                "tool_name": spec.name,
                "version": spec.version,
            }

            stats = self._telemetry.get_tool_stats(spec.name)
            invoked = stats.get("tool_invoked", 0)
            succeeded = stats.get("tool_succeeded", 0)
            failed = stats.get("tool_failed", 0)

            detail["invoked"] = invoked
            detail["succeeded"] = succeeded
            detail["failed"] = failed

            # Not enough data yet.
            if invoked == 0:
                detail["action"] = "skipped"
                detail["reason"] = "No invocations recorded"
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            success_rate = succeeded / invoked if invoked > 0 else 0.0
            detail["success_rate"] = round(success_rate, 4)

            if succeeded < min_success_count:
                detail["action"] = "skipped"
                detail["reason"] = (
                    f"succeeded ({succeeded}) < min_success_count ({min_success_count})"
                )
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            if success_rate < min_success_rate:
                detail["action"] = "skipped"
                detail["reason"] = (
                    f"success_rate ({success_rate:.4f}) < "
                    f"min_success_rate ({min_success_rate})"
                )
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            # Promote: update DB status and refresh cache.
            try:
                conn = self._registry._conn
                if conn is not None:
                    conn.execute(
                        "UPDATE tools SET status = 'active', updated_at = ? "
                        "WHERE name = ? AND version = ? AND status = 'candidate'",
                        (now, spec.name, spec.version),
                    )
                    conn.commit()
                self._registry._load_cache()
            except Exception as exc:
                logger.error("Failed to promote '%s': %s", spec.name, exc)
                detail["action"] = "error"
                detail["reason"] = str(exc)
                result["skipped"] += 1
                result["details"].append(detail)
                continue

            # Telemetry event.
            self._telemetry.record(
                "tool_promoted",
                tool_name=spec.name,
                tool_version=spec.version,
                metadata={
                    "invoked": invoked,
                    "succeeded": succeeded,
                    "success_rate": round(success_rate, 4),
                },
            )

            detail["action"] = "promoted"
            detail["new_status"] = "active"
            result["promoted"] += 1
            result["details"].append(detail)
            logger.info(
                "Promoted '%s' v%s to ACTIVE (succeeded=%d, rate=%.4f)",
                spec.name,
                spec.version,
                succeeded,
                success_rate,
            )

        return result

    def compute_metrics(self) -> dict[str, Any]:
        """Compute system-wide tool evolution metrics.

        Delegates to the metrics computation logic (imported lazily to avoid
        circular dependencies).
        """
        return _compute_metrics_impl(self._registry, self._telemetry)

    async def run_once(self, dry_run: bool = True) -> dict[str, Any]:
        """Run all evolution steps in sequence.

        Order: requests -> absorber -> promote -> metrics.

        Returns a combined dict with results from each step.
        """
        combined: dict[str, Any] = {}

        # Step 1: Process pending tool requests.
        try:
            combined["requests"] = await self.process_pending_requests(
                limit=10, dry_run=dry_run
            )
        except Exception as exc:
            logger.error("process_pending_requests failed: %s", exc)
            combined["requests"] = {"error": str(exc)}

        # Step 2: Run absorber.
        try:
            combined["absorber"] = await self.run_absorber(
                dry_run=dry_run, max_clusters=5
            )
        except Exception as exc:
            logger.error("run_absorber failed: %s", exc)
            combined["absorber"] = {"error": str(exc)}

        # Step 3: Promote candidates.
        try:
            combined["promote"] = self.promote_candidates(
                min_success_count=3, min_success_rate=0.8
            )
        except Exception as exc:
            logger.error("promote_candidates failed: %s", exc)
            combined["promote"] = {"error": str(exc)}

        # Step 4: Compute metrics.
        try:
            combined["metrics"] = self.compute_metrics()
        except Exception as exc:
            logger.error("compute_metrics failed: %s", exc)
            combined["metrics"] = {"error": str(exc)}

        return combined

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_detail_from_row(row: dict[str, Any]) -> dict[str, Any]:
        """Extract a detail dict from a tool_requests DB row."""
        return {
            "request_id": row.get("id"),
            "candidate_name": row.get("candidate_name", ""),
            "risk_level": row.get("risk_level", "L0"),
            "missing_capability": row.get("missing_capability", ""),
            "reason": row.get("reason", ""),
        }

    @staticmethod
    def _row_to_tool_request(row: dict[str, Any]) -> ToolRequest:
        """Convert a tool_requests DB row back to a ToolRequest instance."""
        try:
            risk_level = RiskLevel(row.get("risk_level", "L0"))
        except ValueError:
            risk_level = RiskLevel.L0

        return ToolRequest(
            task_id=row.get("task_id", ""),
            session_id=row.get("session_id", ""),
            reason=row.get("reason", ""),
            missing_capability=row.get("missing_capability", ""),
            candidate_name=row.get("candidate_name", ""),
            candidate_description=row.get("candidate_desc", ""),
            candidate_input_schema=json.loads(row.get("candidate_input") or "{}"),
            candidate_output_schema=json.loads(row.get("candidate_output") or "{}"),
            risk_level=risk_level,
            privacy_notes=row.get("privacy_notes", ""),
            metadata=json.loads(row.get("metadata") or "{}"),
            created_at=row.get("created_at", ""),
        )


# ------------------------------------------------------------------
# Metrics computation (standalone to allow lazy import)
# ------------------------------------------------------------------


def _compute_metrics_impl(registry: Any, telemetry: Any) -> dict[str, Any]:
    """Compute system-wide tool evolution metrics.

    Gathers counts from the registry, request stats from telemetry,
    and per-tool success rates.
    """
    all_tools = registry.list_tools(status=None)

    by_status: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    for spec in all_tools:
        by_status[spec.status.value] = by_status.get(spec.status.value, 0) + 1
        by_risk[spec.risk_level.value] = by_risk.get(spec.risk_level.value, 0) + 1

    # Tool request stats from telemetry.
    request_stats: dict[str, int] = {}
    try:
        request_stats = telemetry.get_tool_request_stats()
    except Exception:
        pass

    # Per-tool success rates for generated tools.
    generated_stats: list[dict[str, Any]] = []
    for spec in all_tools:
        if spec.runtime.value == "python_generated":
            stats = telemetry.get_tool_stats(spec.name)
            invoked = stats.get("tool_invoked", 0)
            succeeded = stats.get("tool_succeeded", 0)
            if invoked > 0:
                generated_stats.append(
                    {
                        "name": spec.name,
                        "status": spec.status.value,
                        "invoked": invoked,
                        "succeeded": succeeded,
                        "failed": stats.get("tool_failed", 0),
                        "success_rate": round(succeeded / invoked, 4),
                    }
                )

    return {
        "total_tools": len(all_tools),
        "by_status": by_status,
        "by_risk_level": by_risk,
        "tool_requests": request_stats,
        "generated_tools": generated_stats,
        "computed_at": datetime.now(timezone.utc).isoformat(),
    }
