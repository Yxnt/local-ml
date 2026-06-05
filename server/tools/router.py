"""Runtime executors — dispatch a ToolSpec + arguments to the right backend.

Each executor handles one ``ToolRuntime`` variant.  The ``ToolRuntimeRouter``
selects the correct executor based on ``ToolSpec.runtime``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from server.tools.spec import (
    RiskLevel,
    ToolContext,
    ToolResult,
    ToolRuntime,
    ToolSpec,
)
from server.tools.telemetry import TelemetryService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Path to the subprocess runner
# ---------------------------------------------------------------------------

_SANDBOX_RUNNER = Path(__file__).parent / "sandbox_runner.py"


# ---------------------------------------------------------------------------
# Clean environment
# ---------------------------------------------------------------------------

_SENSITIVE_KEYWORDS = frozenset({"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"})


def _clean_env() -> dict[str, str]:
    """Build an env dict that strips sensitive variables.

    Also overrides HOME to a temp dir so generated tools cannot read
    user dotfiles.  PATH and PYTHONPATH are preserved.
    """
    clean = {
        k: v
        for k, v in os.environ.items()
        if not any(kw in k.upper() for kw in _SENSITIVE_KEYWORDS)
    }
    clean["HOME"] = tempfile.mkdtemp(prefix="sandbox_home_")
    return clean


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Base class for runtime-specific executors."""

    async def execute(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Memory tools
# ---------------------------------------------------------------------------


class MemoryToolExecutor(ToolExecutor):
    """Executes tools owned by MemoryManager."""

    def __init__(self, memory_manager: Any) -> None:
        self._mm = memory_manager

    async def execute(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = spec.name
        try:
            if name == "memory_remember":
                from memory.memory import MemoryType

                content = arguments.get("content", "")
                mtype_str = arguments.get("type", "fact")
                importance = arguments.get("importance", 0.5)
                mtype = MemoryType(mtype_str) if mtype_str in [e.value for e in MemoryType] else MemoryType.FACT
                mid = self._mm.remember(content, mtype, importance)
                return ToolResult(content=json.dumps({"status": "ok", "id": mid}))

            elif name == "memory_recall":
                query = arguments.get("query", "")
                limit = arguments.get("limit", 10)
                mtype_str = arguments.get("type")
                from memory.memory import MemoryType

                mtype = MemoryType(mtype_str) if mtype_str and mtype_str in [e.value for e in MemoryType] else None
                results = self._mm.recall(query, mtype, limit)
                return ToolResult(content=json.dumps(results, ensure_ascii=False))

            elif name == "memory_stats":
                stats = self._mm.store.get_stats()
                return ToolResult(content=json.dumps(stats, ensure_ascii=False))

            else:
                return ToolResult(content=f"未知 memory 工具: {name}", success=False, error_type="unknown_tool")

        except Exception as e:
            return ToolResult(content=f"memory 工具调用失败: {e}", success=False, error_type=type(e).__name__)


# ---------------------------------------------------------------------------
# Integration tools
# ---------------------------------------------------------------------------


class IntegrationToolExecutor(ToolExecutor):
    """Executes tools owned by Integration instances."""

    def __init__(self, get_integration: Any) -> None:
        """``get_integration(name)`` returns the Integration instance or None."""
        self._get = get_integration
        # tool_name -> (integration_name, method_name, arg_keys, defaults)
        self._routes: dict[str, tuple[str, str, list[str], dict[str, Any]]] = {
            "obsidian_search": ("obsidian", "query", ["query", "limit"], {"limit": 10}),
            "obsidian_read": ("obsidian", "read_note", ["path"], {}),
            "calendar_search": ("calendar", "query", ["query"], {}),
            "calendar_upcoming": ("calendar", "get_upcoming", ["days"], {"days": 7}),
            "email_search": ("email", "query", ["query", "limit"], {"limit": 10}),
            "email_recent": ("email", "get_recent", ["limit"], {"limit": 20}),
            "photos_search": ("photos", "handle_search", ["query", "date_from", "date_to", "album", "limit"], {"limit": 10}),
            "photos_list_albums": ("photos", "handle_list_albums", [], {}),
            "photos_describe": ("photos", "handle_describe", ["photo_id", "question"], {}),
            "smart_home_list_devices": ("smarthome", "list_devices", [], {}),
            "smart_home_control": ("smarthome", "control", ["device_id", "action", "params"], {}),
        }

    async def execute(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        name = spec.name
        route = self._routes.get(name)
        if route is None:
            return ToolResult(content=f"未知集成工具: {name}", success=False, error_type="unknown_tool")

        integ_name, method_name, arg_keys, defaults = route
        integ = self._get(integ_name)
        if integ is None:
            return ToolResult(
                content=f"{integ_name} 未连接",
                success=False,
                error_type="not_connected",
            )

        method = getattr(integ, method_name, None)
        if method is None:
            return ToolResult(
                content=f"{integ_name} 不支持 {method_name}",
                success=False,
                error_type="method_missing",
            )

        # Build keyword args: use provided values, fall back to defaults
        call_args = {k: arguments[k] for k in arg_keys if k in arguments}
        for k, v in defaults.items():
            call_args.setdefault(k, v)

        try:
            result = await method(**call_args)
            return ToolResult(content=json.dumps(result, ensure_ascii=False, default=str))
        except Exception as e:
            return ToolResult(
                content=f"{integ_name} 工具调用失败: {e}",
                success=False,
                error_type=type(e).__name__,
            )


# ---------------------------------------------------------------------------
# Computer-use tools
# ---------------------------------------------------------------------------


class ComputerUseToolExecutor(ToolExecutor):
    """Delegates to ComputerUseAgent."""

    def __init__(self, model_registry: Any) -> None:
        self._registry = model_registry

    async def execute(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        action = arguments.get("action")
        if not action:
            return ToolResult(content="缺少 action 参数", success=False, error_type="missing_arg")

        try:
            from computer_use.agent import ComputerUseAgent

            backend = self._registry.get_backend(ctx.model_name or "gemma-4-e2b-it-4bit")
            cu_agent = ComputerUseAgent(backend, verbose=False)
            result = await cu_agent.run(f"执行操作: {action}，参数: {json.dumps(arguments, ensure_ascii=False)}")
            return ToolResult(content=result)
        except ImportError:
            return ToolResult(
                content="computer_use 模块不可用",
                success=False,
                error_type="import_error",
            )
        except Exception as e:
            return ToolResult(
                content=f"桌面操作失败: {e}",
                success=False,
                error_type=type(e).__name__,
            )


# ---------------------------------------------------------------------------
# Generated Python tool executor
# ---------------------------------------------------------------------------


class GeneratedPythonExecutor(ToolExecutor):
    """Executes sandboxed generated Python tools.

    By default (*execution_mode="subprocess"*) the tool runs in a child
    process via ``sandbox_runner.py``.  Set *execution_mode="in_process_dev_only"*
    to load and run in the current process — useful for debugging during
    development but **not safe for production**.
    """

    def __init__(
        self,
        sandbox_dir: str = "server/tools/sandbox",
        execution_mode: str = "subprocess",
    ) -> None:
        if execution_mode not in ("subprocess", "in_process_dev_only"):
            raise ValueError(
                f"execution_mode must be 'subprocess' or 'in_process_dev_only', got {execution_mode!r}"
            )
        self._sandbox_dir = sandbox_dir
        self._execution_mode = execution_mode

    async def execute(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        if self._execution_mode == "in_process_dev_only":
            return self._execute_in_process(spec, arguments)
        return self._execute_subprocess(spec, arguments)

    # -- subprocess execution ------------------------------------------------

    def _execute_subprocess(self, spec: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
        """Run the tool in an isolated child process."""
        sandbox_path = Path(self._sandbox_dir).resolve()
        sandbox_path.mkdir(parents=True, exist_ok=True)

        file_path = sandbox_path / f"{spec.name}.py"
        if not file_path.exists():
            return ToolResult(
                content=f"Generated tool file not found: {file_path}",
                success=False,
                error_type="file_not_found",
            )

        # Write arguments to a temp file so they never touch stdin quoting
        # issues and can be inspected for debugging.
        args_fd: int | None = None
        args_path: str = ""
        try:
            args_fd, args_path = tempfile.mkstemp(
                suffix=".json", prefix="tool_args_", dir=str(sandbox_path)
            )
            with os.fdopen(args_fd, "w", encoding="utf-8") as fh:
                json.dump(arguments, fh, ensure_ascii=False)
                args_fd = None  # fdopen takes ownership

            config = json.dumps(
                {
                    "tool_name": spec.name,
                    "source_code": "",  # already on disk
                    "arguments_file": args_path,
                    "mode": "execute",
                    "sandbox_dir": str(sandbox_path),
                    "timeout": 15,
                },
                ensure_ascii=False,
            )

            try:
                proc = subprocess.run(
                    [sys.executable, str(_SANDBOX_RUNNER)],
                    input=config,
                    capture_output=True,
                    timeout=20,
                    text=True,
                    env=_clean_env(),
                )
            except subprocess.TimeoutExpired:
                return ToolResult(
                    content=f"Generated tool '{spec.name}' timed out (20 s)",
                    success=False,
                    error_type="timeout",
                )

            # Non-zero exit with no stdout means the runner itself crashed
            if proc.returncode != 0 and not proc.stdout.strip():
                stderr_snippet = (proc.stderr or "").strip()[-500:]
                return ToolResult(
                    content=f"Generated tool '{spec.name}' crashed (exit {proc.returncode}): {stderr_snippet}",
                    success=False,
                    error_type="subprocess_error",
                )

            # Parse JSON result
            try:
                output = json.loads(proc.stdout.strip())
            except json.JSONDecodeError as exc:
                stderr_snippet = (proc.stderr or "").strip()[-500:]
                return ToolResult(
                    content=(
                        f"Invalid JSON from tool '{spec.name}': {exc}\n"
                        f"stdout: {proc.stdout[:300]}\n"
                        f"stderr: {stderr_snippet}"
                    ),
                    success=False,
                    error_type="json_error",
                )

            if output.get("success"):
                result_dict = output.get("result") or {}
                return ToolResult(
                    content=json.dumps(result_dict, ensure_ascii=False, default=str)
                )
            else:
                errors = output.get("errors", [])
                return ToolResult(
                    content=f"Generated tool '{spec.name}' failed: {'; '.join(str(e) for e in errors)}",
                    success=False,
                    error_type="execution_error",
                )

        finally:
            # Clean up the temp args file
            if args_path and os.path.exists(args_path):
                try:
                    os.unlink(args_path)
                except OSError:
                    pass

    # -- in-process execution (dev only) -------------------------------------

    def _execute_in_process(self, spec: ToolSpec, arguments: dict[str, Any]) -> ToolResult:
        """Load and run the tool in the current process (development only)."""
        import importlib
        import importlib.util

        file_path = Path(self._sandbox_dir) / f"{spec.name}.py"
        if not file_path.exists():
            return ToolResult(
                content=f"Generated tool file not found: {file_path}",
                success=False,
                error_type="file_not_found",
            )

        try:
            module_name = f"sandbox_{spec.name}"
            mod_spec = importlib.util.spec_from_file_location(module_name, file_path)
            if mod_spec is None or mod_spec.loader is None:
                return ToolResult(
                    content=f"Cannot load module spec for {spec.name}",
                    success=False,
                    error_type="import_error",
                )

            module = importlib.util.module_from_spec(mod_spec)
            sandbox_str = str(Path(self._sandbox_dir).resolve())
            added = False
            if sandbox_str not in sys.path:
                sys.path.insert(0, sandbox_str)
                added = True
            try:
                mod_spec.loader.exec_module(module)
            finally:
                if added and sandbox_str in sys.path:
                    sys.path.remove(sandbox_str)

            input_instance = module.InputModel(**arguments)
            output = module.run(input_instance)

            if hasattr(output, "model_dump"):
                result_dict = output.model_dump()
            elif hasattr(output, "dict"):
                result_dict = output.dict()
            else:
                result_dict = vars(output)

            return ToolResult(content=json.dumps(result_dict, ensure_ascii=False, default=str))

        except Exception as e:
            return ToolResult(
                content=f"Generated tool '{spec.name}' failed: {e}",
                success=False,
                error_type=type(e).__name__,
            )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


class ToolRuntimeRouter:
    """Selects the right executor for a ToolSpec and runs it.

    Also emits telemetry events around every execution.
    """

    def __init__(
        self,
        memory_executor: MemoryToolExecutor,
        integration_executor: IntegrationToolExecutor,
        computer_executor: ComputerUseToolExecutor,
        generated_executor: GeneratedPythonExecutor | None = None,
        telemetry: TelemetryService | None = None,
    ) -> None:
        self._executors: dict[ToolRuntime, ToolExecutor] = {
            ToolRuntime.MEMORY_METHOD: memory_executor,
            ToolRuntime.INTEGRATION_METHOD: integration_executor,
            ToolRuntime.COMPUTER_USE: computer_executor,
            ToolRuntime.PYTHON_GENERATED: generated_executor or GeneratedPythonExecutor(),
        }
        self._telemetry = telemetry

    async def dispatch(self, spec: ToolSpec, arguments: dict[str, Any], ctx: ToolContext) -> ToolResult:
        executor = self._executors.get(spec.runtime)
        if executor is None:
            return ToolResult(
                content=f"No executor for runtime: {spec.runtime.value}",
                success=False,
                error_type="no_executor",
            )

        start = time.monotonic()
        if self._telemetry:
            self._telemetry.record_tool_invoked(
                spec.name,
                args=arguments,
                task_id=ctx.task_id,
                session_id=ctx.session_id,
                tool_version=spec.version,
            )

        result = await executor.execute(spec, arguments, ctx)
        latency_ms = int((time.monotonic() - start) * 1000)

        if self._telemetry:
            if result.success:
                self._telemetry.record_tool_succeeded(
                    spec.name,
                    result_summary=result.content[:200],
                    latency_ms=latency_ms,
                    task_id=ctx.task_id,
                    session_id=ctx.session_id,
                    tool_version=spec.version,
                )
            else:
                self._telemetry.record_tool_failed(
                    spec.name,
                    error_type=result.error_type or "unknown",
                    error_message=result.content[:500],
                    latency_ms=latency_ms,
                    task_id=ctx.task_id,
                    session_id=ctx.session_id,
                    tool_version=spec.version,
                )

        return result
