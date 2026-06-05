"""Tests for server.tools.orchestrator — ToolEvolutionOrchestrator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from server.tools.spec import RiskLevel, ToolRuntime, ToolSpec, ToolStatus
from server.tools.orchestrator import ToolEvolutionOrchestrator


def _make_orchestrator(registry, telemetry):
    """Build an orchestrator with mock developer/verifier/absorber."""
    developer = MagicMock()
    verifier = MagicMock()
    absorber = MagicMock()
    absorber.run = AsyncMock(return_value={
        "clusters_found": 0,
        "clusters_merged": 0,
        "tools_deprecated": 0,
        "details": [],
    })
    return ToolEvolutionOrchestrator(
        registry=registry,
        telemetry=telemetry,
        developer=developer,
        verifier=verifier,
        absorber=absorber,
    )


class TestProcessPendingRequestsNoRequests:
    """process_pending_requests with no requests returns empty."""

    @pytest.mark.asyncio
    async def test_no_requests_returns_empty(self, registry, telemetry):
        orch = _make_orchestrator(registry, telemetry)
        result = await orch.process_pending_requests(limit=5, dry_run=False)
        assert result["processed"] == 0
        assert result["skipped"] == 0
        assert result["errors"] == 0
        assert result["details"] == []


class TestProcessPendingRequestsL2Skipped:
    """process_pending_requests with L2+ request skips it."""

    @pytest.mark.asyncio
    async def test_l2_request_skipped(self, registry, telemetry):
        """Insert a tool_request with L2 risk, verify it's skipped."""
        from server.tools.spec import ToolRequest

        req = ToolRequest(
            task_id="t1",
            session_id="s1",
            reason="need network access",
            missing_capability="fetch_url",
            candidate_name="url_fetcher",
            candidate_description="Fetch a URL",
            candidate_input_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.L2,
        )
        telemetry.record_tool_request(req)

        orch = _make_orchestrator(registry, telemetry)
        result = await orch.process_pending_requests(limit=5, dry_run=False)
        assert result["processed"] == 0
        assert result["skipped"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["action"] == "skipped"
        assert "L2" in result["details"][0]["reason"]


class TestProcessPendingRequestsDryRun:
    """process_pending_requests dry_run doesn't register."""

    @pytest.mark.asyncio
    async def test_dry_run_does_not_register(self, registry, telemetry):
        from server.tools.spec import ToolRequest

        req = ToolRequest(
            task_id="t1",
            session_id="s1",
            reason="need calculator",
            missing_capability="calculate",
            candidate_name="calc_tool",
            candidate_description="A calculator",
            candidate_input_schema={"type": "object", "properties": {}},
            risk_level=RiskLevel.L0,
        )
        telemetry.record_tool_request(req)

        orch = _make_orchestrator(registry, telemetry)
        result = await orch.process_pending_requests(limit=5, dry_run=True)
        assert result["processed"] == 1
        assert result["details"][0]["action"] == "dry_run"

        # Tool should NOT be in the registry
        assert registry.get_tool("calc_tool") is None


class TestPromoteCandidatesInsufficientStats:
    """promote_candidates with insufficient stats doesn't promote."""

    def test_no_candidates_returns_empty(self, registry, telemetry):
        orch = _make_orchestrator(registry, telemetry)
        result = orch.promote_candidates()
        assert result["promoted"] == 0
        assert result["skipped"] == 0

    def test_candidate_with_no_invocations_skipped(self, registry, telemetry):
        spec = ToolSpec(
            name="gen_tool",
            description="A generated tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.PYTHON_GENERATED,
            status=ToolStatus.CANDIDATE,
        )
        registry.register(spec)

        orch = _make_orchestrator(registry, telemetry)
        result = orch.promote_candidates(min_success_count=3, min_success_rate=0.8)
        assert result["promoted"] == 0
        assert result["skipped"] == 1
        assert result["details"][0]["action"] == "skipped"
        assert "No invocations" in result["details"][0]["reason"]

    def test_candidate_below_min_success_count_skipped(self, registry, telemetry):
        spec = ToolSpec(
            name="gen_tool",
            description="A generated tool",
            input_schema={"type": "object", "properties": {}},
            runtime=ToolRuntime.PYTHON_GENERATED,
            status=ToolStatus.CANDIDATE,
        )
        registry.register(spec)

        # Record 2 invocations, 2 successes (below min_success_count=3)
        telemetry.record_tool_invoked("gen_tool")
        telemetry.record_tool_succeeded("gen_tool")
        telemetry.record_tool_invoked("gen_tool")
        telemetry.record_tool_succeeded("gen_tool")

        orch = _make_orchestrator(registry, telemetry)
        result = orch.promote_candidates(min_success_count=3, min_success_rate=0.8)
        assert result["promoted"] == 0
        assert result["skipped"] == 1
        assert "min_success_count" in result["details"][0]["reason"]
