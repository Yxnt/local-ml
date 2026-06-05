"""Bootstrap — register all existing tools into the ToolRegistry.

Called once during Agent initialisation.  Each section converts an existing
component's tool definitions into ``ToolSpec`` objects and registers them.
"""

from __future__ import annotations

import logging
from typing import Any

from server.tools.registry import ToolRegistry
from server.tools.spec import RiskLevel, ToolRuntime, ToolSpec

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _openai_to_spec(
    oa: dict[str, Any],
    runtime: ToolRuntime,
    provider: str = "",
    risk_level: RiskLevel = RiskLevel.L0,
    entrypoint: str = "",
) -> ToolSpec:
    """Convert an OpenAI function-calling dict to a ToolSpec."""
    fn = oa.get("function", oa)
    return ToolSpec(
        name=fn["name"],
        description=fn.get("description", ""),
        input_schema=fn.get("parameters", {"type": "object", "properties": {}}),
        runtime=runtime,
        provider=provider,
        risk_level=risk_level,
        entrypoint=entrypoint,
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_memory_tools(registry: ToolRegistry, memory_manager: Any) -> None:
    """Register the 3 memory tools."""
    for oa in memory_manager.get_tools():
        spec = _openai_to_spec(
            oa,
            runtime=ToolRuntime.MEMORY_METHOD,
            provider="memory",
            risk_level=RiskLevel.L0,
        )
        registry.register(spec)
    logger.info("Registered memory tools")


def register_integration_tools(registry: ToolRegistry, name: str, integration: Any) -> None:
    """Register tools from a single Integration instance."""
    risk_map = {
        "obsidian": RiskLevel.L2,
        "calendar": RiskLevel.L2,
        "email": RiskLevel.L2,
        "photos": RiskLevel.L2,
        "smarthome": RiskLevel.L4,
    }
    risk = risk_map.get(name, RiskLevel.L2)

    if hasattr(integration, "get_tools"):
        try:
            tools_list = integration.get_tools()
        except Exception:
            logger.warning("Failed to get tools from %s", name, exc_info=True)
            return

        # Guard against coroutine (async get_tools called from sync context)
        import inspect
        if inspect.iscoroutine(tools_list):
            logger.warning("get_tools() for %s returned coroutine; skipping (use async path)", name)
            tools_list.close()
            return

        for oa in tools_list:
            spec = _openai_to_spec(
                oa,
                runtime=ToolRuntime.INTEGRATION_METHOD,
                provider=name,
                risk_level=risk,
            )
            registry.register(spec)
        logger.info("Registered %s tools", name)


def register_computer_use_tool(registry: ToolRegistry) -> None:
    """Register the computer_action tool."""
    from computer_use.tools import COMPUTER_USE_TOOL

    spec = _openai_to_spec(
        COMPUTER_USE_TOOL,
        runtime=ToolRuntime.COMPUTER_USE,
        provider="computer_use",
        risk_level=RiskLevel.L4,
    )
    registry.register(spec)
    logger.info("Registered computer_use tool")


def bootstrap_all_tools(
    registry: ToolRegistry,
    memory_manager: Any,
    integrations: dict[str, Any] | None = None,
    include_computer_use: bool = True,
) -> None:
    """One-call bootstrap: register every known tool."""
    register_memory_tools(registry, memory_manager)

    if integrations:
        for name, integ in integrations.items():
            register_integration_tools(registry, name, integ)

    if include_computer_use:
        try:
            register_computer_use_tool(registry)
        except Exception as e:
            logger.warning("Failed to register computer_use tool: %s", e)

    count = len(registry.list_tools())
    logger.info("Bootstrap complete: %d tools registered", count)
