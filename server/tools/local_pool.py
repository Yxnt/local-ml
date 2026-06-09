"""Query-local tool overlay for in-situ generated tools."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from server.tools.router import ToolRuntimeRouter
from server.tools.spec import ToolContext, ToolResult, ToolRuntime, ToolSpec, ToolStatus


class LocalToolPool:
    """Holds generated tools for one query/session.

    The durable registry remains the global pool. This overlay is local-first
    for lookup and listing, and writes generated source into the sandbox layout
    expected by ``GeneratedPythonExecutor``.
    """

    def __init__(
        self,
        router: ToolRuntimeRouter,
        sandbox_dir: str,
        global_registry: Any = None,
    ) -> None:
        self._router = router
        self._sandbox_dir = Path(sandbox_dir)
        self._global_registry = global_registry
        self._tools: dict[str, ToolSpec] = {}

    @property
    def sandbox_dir(self) -> Path:
        return self._sandbox_dir

    def register(self, spec: ToolSpec, source_code: str) -> ToolSpec:
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)
        source_file = self._source_file_for(spec.name)
        local_spec = replace(
            spec,
            runtime=ToolRuntime.PYTHON_GENERATED,
            provider="query_local",
            status=ToolStatus.ACTIVE,
            metadata={
                **spec.metadata,
                "local_pool": True,
                "source_file": str(source_file),
            },
        )
        source_file.write_text(source_code, encoding="utf-8")
        self._tools[local_spec.name] = local_spec
        return local_spec

    def get_tool(self, name: str) -> ToolSpec | None:
        local = self._tools.get(name)
        if local is not None:
            return local
        if self._global_registry is not None:
            return self._global_registry.get_tool(name)
        return None

    def list_specs(self) -> list[ToolSpec]:
        return list(self._tools.values())

    def list_openai_tools(self, global_tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        tools = [spec.to_openai_schema() for spec in self._tools.values()]
        seen = {tool["function"]["name"] for tool in tools}

        if global_tools is None and self._global_registry is not None:
            global_tools = self._global_registry.list_openai_tools()

        for tool in global_tools or []:
            name = tool.get("function", {}).get("name")
            if name and name not in seen:
                tools.append(tool)
                seen.add(name)
        return tools

    async def dispatch(self, name: str, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        spec = self._tools.get(name)
        if spec is None:
            return ToolResult(
                content=f"未知工具: {name}",
                success=False,
                error_type="tool_not_found",
            )
        return await self._router.dispatch(spec, arguments, ctx)

    def _source_file_for(self, tool_name: str) -> Path:
        source_file = self._sandbox_dir / f"{tool_name}.py"
        if source_file.parent != self._sandbox_dir:
            raise ValueError(f"Invalid local tool name: {tool_name!r}")
        return source_file
