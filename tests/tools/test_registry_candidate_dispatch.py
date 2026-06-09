from __future__ import annotations

import pytest

from server.tools.registry import ToolRegistry
from server.tools.spec import RiskLevel, ToolContext, ToolRuntime, ToolSpec, ToolStatus


@pytest.mark.asyncio
async def test_candidate_tool_is_not_dispatchable(tmp_path):
    registry = ToolRegistry(db_path=str(tmp_path / "tools.db"))
    registry.connect()
    try:
        registry.register(
            ToolSpec(
                name="candidate_reverse",
                description="Candidate only",
                input_schema={"type": "object", "properties": {}},
                runtime=ToolRuntime.PYTHON_GENERATED,
                provider="query_local_candidate",
                risk_level=RiskLevel.L0,
                status=ToolStatus.CANDIDATE,
            )
        )

        result = await registry.dispatch("candidate_reverse", {}, ToolContext())

        assert result.success is False
        assert result.error_type == "tool_candidate"
    finally:
        registry.disconnect()
