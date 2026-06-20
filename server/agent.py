"""Core Agent - ties together model, memory, tools, and integrations.

Usage:
    agent = Agent()
    response = await agent.run("Hello!")
    await agent.close()
"""

from __future__ import annotations

import json
import logging
import os
import sys
import uuid
from dataclasses import dataclass, field
from typing import Any

# Ensure project root is on sys.path before any imports of sibling packages.
import pathlib
_PROJECT_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
    # Force optimization to be importable
    import optimization.collector  # noqa: F401

from backends.registry import ModelRegistry
from memory.manager import MemoryManager
from optimization.collector import UsageCollector, Outcome
from personal_evolution.context import build_approved_memory_context
from personal_evolution.store import PersonalEvolutionStore
from server.context_manager import ContextManager
from server.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

# -- Tool dispatch table (legacy) ------------------------------------------
_MEMORY_TOOLS = {"memory_remember", "memory_recall", "memory_stats"}
_OBSIDIAN_TOOLS = {"obsidian_search", "obsidian_read"}
_CALENDAR_TOOLS = {"calendar_search", "calendar_upcoming"}
_EMAIL_TOOLS = {"email_search", "email_recent"}
_COMPUTER_TOOLS = {"computer_action"}

_INTEGRATION_ERRORS: dict[str, str] = {
    "obsidian": "请先配置 Obsidian 路径（在 config.yaml 的 integrations.obsidian.vaults 中设置）",
    "calendar": "请先配置日历账号（在 config.yaml 的 integrations.calendar.accounts 中设置）",
    "email": "请先配置邮箱账号（在 config.yaml 的 integrations.email.accounts 中设置）",
    "computer": "请先配置电脑控制功能（computer_use 模块）",
}


@dataclass
class _ToolDispatchContext:
    task_id: str
    user_input: str
    local_tool_pool: Any = None
    attempted_generation: set[str] = field(default_factory=set)


class Agent:
    """High-level agent that orchestrates model, memory, tools, and integrations.

    Args:
        data_dir: Directory for memory data files (soul, user, db).
        default_model: Model name to use when none is specified.
        max_context_turns: Max conversation turns to keep in context window.
        use_tool_registry: If True, use ToolRegistry + ToolRuntimeRouter for
            dispatch.  Falls back to legacy on init failure.
        tool_retrieval_mode: "legacy" | "all" | "top_k" | "hybrid" | "auto".
        tool_retrieval_top_k: Number of tools to retrieve in top_k mode.
        tool_db_path: SQLite path for ToolRegistry. None = data_dir/usage.db.
        tool_telemetry_db_path: SQLite path for TelemetryService. None = same as tool_db_path.
    """

    def __init__(
        self,
        data_dir: str = "memory/data",
        default_model: str = "gemma-4-e2b-it-4bit",
        max_context_turns: int = 20,
        use_tool_registry: bool = True,
        tool_retrieval_mode: str = "auto",
        tool_retrieval_top_k: int = 12,
        tool_db_path: str | None = None,
        tool_telemetry_db_path: str | None = None,
        enable_in_situ_tool_generation: bool = False,
        tool_sandbox_dir: str | None = None,
        tool_candidate_dir: str | None = None,
        tool_developer: Any = None,
        tool_verifier: Any = None,
        tool_risk_classifier: Any = None,
        tool_local_candidate_exporter: Any = None,
    ) -> None:
        self._data_dir = data_dir
        self._default_model = default_model
        self._current_model = default_model

        # -- Core components --------------------------------------------------
        self._registry = ModelRegistry()
        self._registry.register_defaults()

        self._memory = MemoryManager(data_dir=data_dir)
        self._memory.connect()

        self._context = ContextManager(max_history_turns=max_context_turns)

        self._collector = UsageCollector()
        self._collector.connect()
        self._session_id = self._collector.start_session()

        self._embedding = EmbeddingClient()

        # -- Integrations (lazy-connected) ------------------------------------
        self._integrations: dict[str, Any] = {}
        self._integration_errors: dict[str, str] = {}

        # -- Conversation messages (system + turns) ---------------------------
        self._messages: list[dict[str, Any]] = []

        # -- Tool system (new) ------------------------------------------------
        self._use_tool_registry = use_tool_registry
        self._tool_retrieval_mode = tool_retrieval_mode
        self._tool_retrieval_top_k = tool_retrieval_top_k

        # Tool system objects — set by _init_tool_system or None if legacy.
        self._tool_registry: Any = None
        self._tool_telemetry: Any = None
        self._tool_router: Any = None
        self._tool_retriever: Any = None
        self._tool_db_path: str = tool_db_path or os.path.join(data_dir, "usage.db")
        self._tool_telemetry_db_path: str = tool_telemetry_db_path or self._tool_db_path
        self._enable_in_situ_tool_generation = enable_in_situ_tool_generation
        self._tool_sandbox_dir = tool_sandbox_dir or os.path.join(data_dir, "generated_tools")
        self._tool_candidate_dir = tool_candidate_dir or os.path.join(data_dir, "generated_tool_candidates")
        self._tool_developer = tool_developer
        self._tool_verifier = tool_verifier
        self._tool_risk_classifier = tool_risk_classifier
        self._tool_local_candidate_exporter = tool_local_candidate_exporter

        if self._use_tool_registry:
            self._init_tool_system()

    # ------------------------------------------------------------------
    # Tool system init
    # ------------------------------------------------------------------

    def _init_tool_system(self) -> None:
        """Initialise ToolRegistry, TelemetryService, ToolRuntimeRouter, ToolRetriever.

        On any failure, falls back to legacy mode and logs the error.
        """
        try:
            from server.tools.registry import ToolRegistry
            from server.tools.router import (
                ComputerUseToolExecutor,
                GeneratedPythonExecutor,
                IntegrationToolExecutor,
                MemoryToolExecutor,
                ToolRuntimeRouter,
            )
            from server.tools.telemetry import TelemetryService
            from server.tools.retriever import ToolRetriever

            # Telemetry
            self._tool_telemetry = TelemetryService(db_path=self._tool_telemetry_db_path)
            self._tool_telemetry.connect()

            # Registry
            self._tool_registry = ToolRegistry(
                db_path=self._tool_db_path,
                telemetry=self._tool_telemetry,
            )
            self._tool_registry.connect()

            # Router with executors
            memory_executor = MemoryToolExecutor(self._memory)
            integration_executor = IntegrationToolExecutor(
                lambda name: self._integrations.get(name)
            )
            computer_executor = ComputerUseToolExecutor(self._registry)

            self._tool_router = ToolRuntimeRouter(
                memory_executor=memory_executor,
                integration_executor=integration_executor,
                computer_executor=computer_executor,
                generated_executor=GeneratedPythonExecutor(sandbox_dir=self._tool_sandbox_dir),
                telemetry=self._tool_telemetry,
            )
            self._tool_registry._router = self._tool_router

            # Retriever
            self._tool_retriever = ToolRetriever(
                registry=self._tool_registry,
                embedding_client=self._embedding,
            )

            # Bootstrap existing tools into registry
            self._bootstrap_tool_registry()

            if self._enable_in_situ_tool_generation:
                from server.tools.developer import ToolDeveloper
                from server.tools.local_candidates import LocalCandidateExporter
                from server.tools.risk_classifier import CapabilityRiskClassifier
                from server.tools.verifier import ToolVerifier

                self._tool_developer = self._tool_developer or ToolDeveloper(
                    sandbox_dir=self._tool_sandbox_dir
                )
                self._tool_verifier = self._tool_verifier or ToolVerifier(
                    sandbox_dir=self._tool_sandbox_dir
                )
                self._tool_risk_classifier = (
                    self._tool_risk_classifier or CapabilityRiskClassifier()
                )
                self._tool_local_candidate_exporter = (
                    self._tool_local_candidate_exporter
                    or LocalCandidateExporter(
                        registry=self._tool_registry,
                        telemetry=self._tool_telemetry,
                        candidate_dir=self._tool_candidate_dir,
                    )
                )

            logger.info(
                "Tool system initialised (mode=%s, top_k=%d)",
                self._tool_retrieval_mode,
                self._tool_retrieval_top_k,
            )

        except Exception:
            logger.exception("Failed to initialise tool system, falling back to legacy")
            self._use_tool_registry = False
            self._tool_registry = None
            self._tool_telemetry = None
            self._tool_router = None
            self._tool_retriever = None

    def _bootstrap_tool_registry(self) -> None:
        """Register all existing tools (memory + computer_use) into the registry.

        Integration tools are registered lazily when connect_integration() is called.
        """
        if self._tool_registry is None:
            return

        from server.tools.bootstrap import register_computer_use_tool, register_memory_tools

        register_memory_tools(self._tool_registry, self._memory)

        if "computer" not in self._integration_errors:
            try:
                register_computer_use_tool(self._tool_registry)
            except Exception:
                logger.warning("Failed to register computer_use tool", exc_info=True)

    def _register_integration_tools(self, name: str, integration: Any) -> None:
        """Register tools from a connected integration into the registry."""
        if self._tool_registry is None:
            return
        from server.tools.bootstrap import register_integration_tools

        register_integration_tools(self._tool_registry, name, integration)

    # ------------------------------------------------------------------
    # System prompt
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Assemble system prompt from soul + user + memory + context."""
        parts: list[str] = []

        # Soul + user + recent memories (from MemoryManager).
        memory_prompt = self._memory.get_system_prompt()
        if memory_prompt:
            parts.append(memory_prompt)

        personal_memory_prompt = self._build_personal_evolution_prompt()
        if personal_memory_prompt:
            parts.append(personal_memory_prompt)

        # Contextual entities and recent history.
        context_info = self._context.get_relevant_context("")
        if context_info:
            parts.append(f"\n会话上下文：\n{context_info}")

        # Available integration summary.
        connected = [n for n in self._integrations if n not in self._integration_errors]
        if connected:
            parts.append(f"\n已连接的数据源：{', '.join(connected)}")

        return "\n\n".join(parts)

    def _build_personal_evolution_prompt(self) -> str:
        db_path = os.environ.get("PERSONAL_EVOLUTION_DB")
        if not db_path:
            db_path = os.path.join(self._data_dir, "personal_evolution.sqlite3")
        if not os.path.exists(db_path):
            return ""
        try:
            return build_approved_memory_context(PersonalEvolutionStore(db_path))
        except Exception:
            logger.warning("Failed to load personal evolution memories", exc_info=True)
            return ""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Persist state and release resources."""
        self._collector.end_session()
        self._collector.disconnect()

        if self._tool_registry is not None:
            try:
                self._tool_registry.disconnect()
            except Exception:
                logger.warning("Failed to disconnect tool registry", exc_info=True)

        if self._tool_telemetry is not None:
            try:
                self._tool_telemetry.disconnect()
            except Exception:
                logger.warning("Failed to disconnect tool telemetry", exc_info=True)

        self._memory.disconnect()

        for integ in self._integrations.values():
            if hasattr(integ, "disconnect"):
                try:
                    await integ.disconnect()
                except Exception:
                    pass
        await self._embedding.close()

    async def __aenter__(self) -> Agent:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self, user_input: str, model: str | None = None) -> str:
        """Process user input end-to-end and return the assistant response.

        Steps:
        1. Build system prompt (soul + user + memory).
        2. Append user message to conversation.
        3. Call model to generate a response.
        4. If the response contains tool calls, execute them and loop.
        5. Record the interaction for optimization.
        """
        model_name = model or self._current_model

        # Record task start in telemetry (best-effort).
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        if self._tool_telemetry is not None:
            try:
                self._tool_telemetry.record_task_started(task_id, self._session_id)
            except Exception:
                pass

        # 1. Ensure system prompt is set.
        if not self._messages or self._messages[0].get("role") != "system":
            system_prompt = self._build_system_prompt()
            self._messages.insert(0, {"role": "system", "content": system_prompt})

        # 2. Append user message.
        self._messages.append({"role": "user", "content": user_input})

        # 3. Load model backend.
        backend = await self._registry.get_or_load(model_name)

        # 4. Generate with tool-call loop.
        all_tool_calls: list[dict[str, Any]] = []
        all_tool_results: list[dict[str, Any]] = []
        max_rounds = 5
        dispatch_ctx = _ToolDispatchContext(
            task_id=task_id,
            user_input=user_input,
            local_tool_pool=self._create_local_tool_pool(),
        )

        for _ in range(max_rounds):
            tools = await self.get_tools_async(
                user_input,
                local_tool_pool=dispatch_ctx.local_tool_pool,
            )
            prompt = backend.apply_chat_template(
                self._messages,
                tools=tools if tools else None,
            )
            output_text = backend.generate(prompt=prompt)

            tool_calls = backend.parse_tool_calls(output_text)

            if not tool_calls:
                self._messages.append({"role": "assistant", "content": output_text})
                self._context.update(user_input, output_text)

                self._collector.record_interaction(
                    user_input=user_input,
                    agent_response=output_text,
                    tool_calls=all_tool_calls or None,
                    tool_results=all_tool_results or None,
                    outcome=Outcome.SUCCESS if all_tool_calls else Outcome.UNKNOWN,
                    metadata={"model": model_name, "session_id": self._session_id},
                )

                # Record task finished (best-effort).
                if self._tool_telemetry is not None:
                    try:
                        self._export_local_tool_candidates(dispatch_ctx)
                        self._tool_telemetry.record_task_finished(task_id, self._session_id)
                    except Exception:
                        pass

                return output_text

            # Execute tool calls.
            self._messages.append({
                "role": "assistant",
                "content": output_text,
                "tool_calls": tool_calls,
            })
            all_tool_calls.extend(tool_calls)

            tool_results = await self.execute_tools(tool_calls, dispatch_ctx=dispatch_ctx)
            all_tool_results.extend(tool_results)

            for result in tool_results:
                self._messages.append({
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "name": result["name"],
                    "content": result["content"],
                })

        # Exhausted rounds.
        fallback = "抱歉，工具调用轮次过多，未能完成任务。"
        self._messages.append({"role": "assistant", "content": fallback})
        self._collector.record_interaction(
            user_input=user_input,
            agent_response=fallback,
            tool_calls=all_tool_calls,
            tool_results=all_tool_results,
            outcome=Outcome.FAILURE,
            metadata={"model": model_name, "reason": "max_tool_rounds"},
        )

        if self._tool_telemetry is not None:
            try:
                self._export_local_tool_candidates(dispatch_ctx)
                self._tool_telemetry.record_task_finished(task_id, self._session_id)
            except Exception:
                pass

        return fallback

    async def execute_tools(
        self,
        tool_calls: list[dict[str, Any]],
        dispatch_ctx: _ToolDispatchContext | None = None,
    ) -> list[dict[str, Any]]:
        """Execute tool calls and return results."""
        results: list[dict[str, Any]] = []

        for tc in tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")

            try:
                args = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                args = {}

            try:
                content = await self._dispatch_tool(name, args, dispatch_ctx)
            except Exception as exc:
                logger.exception("Tool %s raised an error", name)
                content = f"工具调用失败: {exc}"

            results.append({
                "tool_call_id": call_id,
                "name": name,
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            })

        return results

    async def get_tools_async(
        self,
        query: str | None = None,
        local_tool_pool: Any = None,
    ) -> list[dict[str, Any]]:
        """Return tool definitions for the current turn.

        Mode ``legacy``: old assembled list.
        Mode ``all``: registry.list_openai_tools().
        Mode ``top_k``: semantic search via ToolRetriever, keyword fallback, then all.
        Mode ``hybrid``: semantic top_k + core tools (memory_stats etc.), deduped.
        Mode ``auto``: all if <= top_k tools, else top_k when query present.
        """
        # Legacy mode — use old get_tools().
        if not self._use_tool_registry or self._tool_retrieval_mode == "legacy":
            return self._legacy_get_tools()

        mode = self._tool_retrieval_mode
        registry = self._tool_registry

        if registry is None:
            return self._legacy_get_tools()

        # Mode: all — return everything from registry.
        if mode == "all":
            return self._with_local_tools(registry.list_openai_tools(), local_tool_pool)

        # Mode: auto — decide based on tool count.
        if mode == "auto":
            all_tools = registry.list_tools()
            if len(all_tools) <= self._tool_retrieval_top_k:
                return self._with_local_tools(registry.list_openai_tools(), local_tool_pool)
            if query:
                mode = "top_k"
            else:
                return self._with_local_tools(registry.list_openai_tools(), local_tool_pool)

        # Mode: top_k — semantic search with fallback.
        if mode == "top_k":
            return self._with_local_tools(
                await self._get_tools_top_k(query),
                local_tool_pool,
            )

        # Mode: hybrid — top_k + core tools.
        if mode == "hybrid":
            return self._with_local_tools(
                await self._get_tools_hybrid(query),
                local_tool_pool,
            )

        # Fallback
        return self._with_local_tools(registry.list_openai_tools(), local_tool_pool)

    def _with_local_tools(
        self,
        global_tools: list[dict[str, Any]],
        local_tool_pool: Any = None,
    ) -> list[dict[str, Any]]:
        if local_tool_pool is None:
            return global_tools
        return local_tool_pool.list_openai_tools(global_tools)

    async def _get_tools_top_k(self, query: str | None) -> list[dict[str, Any]]:
        """Semantic search → keyword fallback → all."""
        registry = self._tool_registry
        retriever = self._tool_retriever

        if retriever is None or not query:
            return registry.list_openai_tools()

        # Semantic search.
        try:
            specs = await retriever.retrieve(query, limit=self._tool_retrieval_top_k)
            if specs:
                return [s.to_openai_schema() for s in specs]
        except Exception:
            logger.warning("Semantic tool retrieval failed, falling back to keyword", exc_info=True)

        # Keyword fallback.
        try:
            keyword_hits = retriever._keyword_search(query, limit=self._tool_retrieval_top_k)
            if keyword_hits:
                return [s.to_openai_schema() for s in keyword_hits]
        except Exception:
            logger.warning("Keyword tool retrieval failed, falling back to all", exc_info=True)

        # All fallback.
        return registry.list_openai_tools()

    async def _get_tools_hybrid(self, query: str | None) -> list[dict[str, Any]]:
        """top_k + core tools, deduplicated."""
        registry = self._tool_registry
        retriever = self._tool_retriever

        core_names = {"memory_remember", "memory_recall", "memory_stats"}
        result_specs: list[Any] = []
        seen: set[str] = set()

        # Add core tools first.
        for name in core_names:
            spec = registry.get_tool(name)
            if spec is not None and spec.name not in seen:
                result_specs.append(spec)
                seen.add(spec.name)

        # Add top-k from retriever.
        if retriever is not None and query:
            try:
                retrieved = await retriever.retrieve(query, limit=self._tool_retrieval_top_k)
                for spec in retrieved:
                    if spec.name not in seen:
                        result_specs.append(spec)
                        seen.add(spec.name)
            except Exception:
                logger.warning("Hybrid retrieval failed, using core + all", exc_info=True)

        if not result_specs:
            return registry.list_openai_tools()

        return [s.to_openai_schema() for s in result_specs]

    def get_tools(self, query: str | None = None) -> list[dict[str, Any]]:
        """Sync wrapper around get_tools_async.

        Falls back to legacy or registry-all when no event loop is running.
        """
        import asyncio

        if not self._use_tool_registry or self._tool_retrieval_mode == "legacy":
            return self._legacy_get_tools()

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # Can't await inside a running loop — fall back to registry-all.
            if self._tool_registry is not None:
                logger.debug("Sync get_tools() inside running loop, returning all registry tools")
                return self._tool_registry.list_openai_tools()
            return self._legacy_get_tools()

        # No running loop: create a short-lived loop explicitly. Python 3.12 no
        # longer guarantees a current loop exists in synchronous test callers.
        return asyncio.run(self.get_tools_async(query))

    def _legacy_get_tools(self) -> list[dict[str, Any]]:
        """Original get_tools logic — assembled from components."""
        tools: list[dict[str, Any]] = []
        tools.extend(self._memory.get_tools())

        for name, integ in self._integrations.items():
            if hasattr(integ, "get_tools"):
                try:
                    tools.extend(integ.get_tools())
                except Exception:
                    logger.warning("Failed to get tools from %s", name)

        if "computer" not in self._integration_errors:
            from computer_use.tools import COMPUTER_USE_TOOL
            tools.append(COMPUTER_USE_TOOL)

        return tools

    async def switch_model(self, model_name: str) -> None:
        """Switch to a different model backend."""
        self._registry.get_backend(model_name)
        await self._registry.get_or_load(model_name)
        self._current_model = model_name
        logger.info("Switched to model: %s", model_name)

    def get_current_model(self) -> str:
        """Return the name of the currently selected model."""
        return self._current_model

    def get_session_id(self) -> str:
        """Return the current session ID."""
        return self._session_id

    def clear_context(self) -> None:
        """Clear conversation history (keeps system prompt)."""
        system = self._messages[0] if self._messages and self._messages[0].get("role") == "system" else None
        self._messages = [system] if system else []
        self._context.context.recent_history.clear()

    # ------------------------------------------------------------------
    # Integration management
    # ------------------------------------------------------------------

    async def connect_integration(self, name: str, integration: Any) -> bool:
        """Attempt to connect an integration.

        On success, registers integration tools into the ToolRegistry (if active).
        Returns True on success, False on failure.
        """
        try:
            if hasattr(integration, "connect"):
                await integration.connect()
            self._integrations[name] = integration
            self._integration_errors.pop(name, None)

            # Register integration tools into the tool registry.
            self._register_integration_tools(name, integration)

            logger.info("Integration connected: %s", name)
            return True
        except Exception as exc:
            self._integration_errors[name] = str(exc)
            logger.warning("Integration %s failed to connect: %s", name, exc)
            return False

    def get_integration_status(self) -> dict[str, dict[str, Any]]:
        """Return status of all known integrations."""
        status: dict[str, dict[str, Any]] = {}
        for name in ("obsidian", "calendar", "email", "computer"):
            if name in self._integrations:
                status[name] = {"connected": True, "error": None}
            elif name in self._integration_errors:
                status[name] = {"connected": False, "error": self._integration_errors[name]}
            else:
                status[name] = {"connected": False, "error": "未初始化"}
        return status

    # ------------------------------------------------------------------
    # Tool dispatch
    # ------------------------------------------------------------------

    async def _dispatch_tool(
        self,
        name: str,
        args: dict[str, Any],
        dispatch_ctx: _ToolDispatchContext | None = None,
    ) -> str:
        """Route a tool call.

        If the tool registry is active, dispatches via registry → router.
        On tool_not_found, records a ToolRequest (dry-run) and returns a
        compatible error string.  On registry failure, falls back to legacy.
        """
        if self._use_tool_registry and self._tool_registry is not None:
            try:
                from server.tools.spec import ToolContext

                ctx = ToolContext(
                    session_id=self._session_id,
                    task_id=dispatch_ctx.task_id if dispatch_ctx else "",
                    user_input=dispatch_ctx.user_input if dispatch_ctx else "",
                    model_name=self._current_model,
                )
                if dispatch_ctx and dispatch_ctx.local_tool_pool is not None:
                    local_result = await dispatch_ctx.local_tool_pool.dispatch(name, args, ctx)
                    if local_result.error_type != "tool_not_found":
                        return local_result.content

                result = await self._tool_registry.dispatch(name, args, ctx)

                if result.error_type == "tool_not_found":
                    # Check if this is a known integration tool — fall back to
                    # legacy dispatch so connected integrations still work even
                    # if their tools weren't registered in the registry.
                    if self._is_integration_tool(name):
                        return await self._legacy_dispatch_tool(name, args)

                    req = self._build_tool_request(name, args, dispatch_ctx)
                    generated = await self._try_in_situ_generation(req, args, ctx, dispatch_ctx)
                    if generated is not None:
                        return generated.content

                    if self._tool_telemetry is not None:
                        try:
                            self._tool_telemetry.record_tool_request(req)
                        except Exception:
                            logger.warning("Failed to record tool request for '%s'", name, exc_info=True)

                    # Return compatible error message.
                    return self._integration_error_for_tool(name)

                return result.content

            except Exception:
                logger.warning("Registry dispatch failed for '%s', falling back to legacy", name, exc_info=True)
                return await self._legacy_dispatch_tool(name, args)

        # Legacy path.
        return await self._legacy_dispatch_tool(name, args)

    def _create_local_tool_pool(self) -> Any:
        if self._tool_router is None:
            return None
        from server.tools.local_pool import LocalToolPool

        return LocalToolPool(
            router=self._tool_router,
            sandbox_dir=self._tool_sandbox_dir,
            global_registry=self._tool_registry,
        )

    def _build_tool_request(
        self,
        name: str,
        args: dict[str, Any],
        dispatch_ctx: _ToolDispatchContext | None,
    ) -> Any:
        from server.tools.spec import ToolRequest

        return ToolRequest(
            task_id=dispatch_ctx.task_id if dispatch_ctx else "",
            session_id=self._session_id,
            reason=f"Model requested tool '{name}' which does not exist",
            missing_capability=f"Tool '{name}' with args {json.dumps(args, ensure_ascii=False)[:200]}",
            candidate_name=name,
            candidate_description=f"Auto-detected need for tool: {name}",
            candidate_input_schema={
                "type": "object",
                "properties": {key: {"type": "string"} for key in args},
                "required": list(args.keys()),
            },
            candidate_output_schema={"type": "object", "properties": {}},
        )

    async def _try_in_situ_generation(
        self,
        request: Any,
        args: dict[str, Any],
        ctx: Any,
        dispatch_ctx: _ToolDispatchContext | None,
    ) -> Any | None:
        if not self._enable_in_situ_tool_generation:
            return None
        if dispatch_ctx is None or dispatch_ctx.local_tool_pool is None:
            return None
        if request.candidate_name in dispatch_ctx.attempted_generation:
            return None
        if (
            self._tool_developer is None
            or self._tool_verifier is None
            or self._tool_risk_classifier is None
        ):
            return None

        dispatch_ctx.attempted_generation.add(request.candidate_name)
        decision = self._tool_risk_classifier.annotate_request(request)
        if self._tool_telemetry is not None:
            try:
                self._tool_telemetry.record(
                    "tool_generation_attempted",
                    tool_name=request.candidate_name,
                    task_id=request.task_id,
                    session_id=request.session_id,
                    metadata={
                        "source": "in_situ_local",
                        "risk_classifier": request.metadata.get("risk_classifier", {}),
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to record generation attempt for '%s'",
                    request.candidate_name,
                    exc_info=True,
                )

        if not decision.auto_generatable:
            return None

        generation = await self._tool_developer.generate(request)
        if not generation.get("success") or not generation.get("source_code"):
            request.metadata["in_situ_generation"] = {
                "success": False,
                "error": generation.get("error"),
            }
            return None

        source_code = generation["source_code"]
        verify_result = self._tool_verifier.verify(request.candidate_name, source_code)
        if not verify_result.passed:
            request.metadata["in_situ_generation"] = {
                "success": False,
                "error": "; ".join(verify_result.errors),
            }
            return None

        spec = self._build_generated_tool_spec(request)
        dispatch_ctx.local_tool_pool.register(spec, source_code)
        request.metadata["in_situ_generation"] = {"success": True}
        if self._tool_telemetry is not None:
            try:
                self._tool_telemetry.record_tool_created(
                    request.candidate_name,
                    metadata={
                        "source": "in_situ_local",
                        "task_id": request.task_id,
                        "session_id": request.session_id,
                        "risk_classifier": request.metadata.get("risk_classifier", {}),
                    },
                )
            except Exception:
                logger.warning(
                    "Failed to record created generated tool '%s'",
                    request.candidate_name,
                    exc_info=True,
                )
        return await dispatch_ctx.local_tool_pool.dispatch(request.candidate_name, args, ctx)

    def _build_generated_tool_spec(self, request: Any) -> Any:
        from server.tools.spec import ToolRuntime, ToolSpec, ToolStatus

        return ToolSpec(
            name=request.candidate_name,
            description=request.candidate_description or request.missing_capability,
            input_schema=request.candidate_input_schema,
            output_schema=request.candidate_output_schema,
            runtime=ToolRuntime.PYTHON_GENERATED,
            provider="query_local",
            risk_level=request.risk_level,
            status=ToolStatus.ACTIVE,
            metadata={
                "generated_from_tool_request": True,
                "task_id": request.task_id,
                "session_id": request.session_id,
                **request.metadata,
            },
        )

    def _export_local_tool_candidates(self, dispatch_ctx: _ToolDispatchContext | None) -> None:
        if dispatch_ctx is None or dispatch_ctx.local_tool_pool is None:
            return
        if self._tool_local_candidate_exporter is None:
            return
        if not dispatch_ctx.local_tool_pool.list_specs():
            return
        self._tool_local_candidate_exporter.export(
            dispatch_ctx.local_tool_pool,
            task_id=dispatch_ctx.task_id,
            batch_id=f"session_{self._session_id}",
        )

    def _integration_error_for_tool(self, name: str) -> str:
        """Return a helpful error message when a tool belongs to a known integration."""
        for integ_name, tools_set in [
            ("obsidian", _OBSIDIAN_TOOLS),
            ("calendar", _CALENDAR_TOOLS),
            ("email", _EMAIL_TOOLS),
            ("computer", _COMPUTER_TOOLS),
        ]:
            if name in tools_set:
                return _INTEGRATION_ERRORS.get(integ_name, f"未知工具: {name}")
        return f"未知工具: {name}"

    def _is_integration_tool(self, name: str) -> bool:
        """Check if a tool name belongs to any known integration."""
        return (
            name in _OBSIDIAN_TOOLS
            or name in _CALENDAR_TOOLS
            or name in _EMAIL_TOOLS
            or name in _COMPUTER_TOOLS
        )

    async def _legacy_dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Original dispatch logic — name-set routing."""
        if name in _MEMORY_TOOLS:
            return self._dispatch_memory_tool(name, args)

        if name in _OBSIDIAN_TOOLS:
            return await self._dispatch_integration_tool(
                "obsidian", name, args,
                error_hint=_INTEGRATION_ERRORS["obsidian"],
            )

        if name in _CALENDAR_TOOLS:
            return await self._dispatch_integration_tool(
                "calendar", name, args,
                error_hint=_INTEGRATION_ERRORS["calendar"],
            )

        if name in _EMAIL_TOOLS:
            return await self._dispatch_integration_tool(
                "email", name, args,
                error_hint=_INTEGRATION_ERRORS["email"],
            )

        if name in _COMPUTER_TOOLS:
            return self._dispatch_computer_tool(args)

        return f"未知工具: {name}"

    def _dispatch_memory_tool(self, name: str, args: dict[str, Any]) -> str:
        """Handle memory_remember, memory_recall, memory_stats."""
        if name == "memory_remember":
            content = args.get("content", "")
            mem_type_str = args.get("type", "fact")
            importance = args.get("importance", 0.5)

            from memory.memory import MemoryType
            try:
                mem_type = MemoryType(mem_type_str)
            except ValueError:
                mem_type = MemoryType.FACT

            memory_id = self._memory.remember(content, mem_type, importance)
            return json.dumps({"status": "ok", "id": memory_id}, ensure_ascii=False)

        if name == "memory_recall":
            query = args.get("query", "")
            limit = args.get("limit", 10)
            mem_type_str = args.get("type")

            from memory.memory import MemoryType
            mem_type = None
            if mem_type_str:
                try:
                    mem_type = MemoryType(mem_type_str)
                except ValueError:
                    pass

            results = self._memory.recall(query, mem_type, limit)
            return json.dumps(results, ensure_ascii=False)

        if name == "memory_stats":
            stats = self._memory.store.get_stats()
            return json.dumps(stats, ensure_ascii=False)

        return f"未知 memory 工具: {name}"

    async def _dispatch_integration_tool(
        self,
        integration_name: str,
        tool_name: str,
        args: dict[str, Any],
        error_hint: str,
    ) -> str:
        """Dispatch a tool call to a named integration."""
        integ = self._integrations.get(integration_name)
        if integ is None:
            return error_hint

        try:
            if tool_name == "obsidian_search":
                query = args.get("query", "")
                limit = args.get("limit", 10)
                results = await integ.query(query, limit)
                return json.dumps(results, ensure_ascii=False)

            if tool_name == "obsidian_read":
                path = args.get("path", "")
                note = await integ.read_note(path)
                if note is None:
                    return f"笔记不存在: {path}"
                return json.dumps(note, ensure_ascii=False)

            if tool_name == "calendar_search":
                query = args.get("query", "")
                results = await integ.query(query)
                return json.dumps(results, ensure_ascii=False)

            if tool_name == "calendar_upcoming":
                days = args.get("days", 7)
                results = await integ.get_upcoming(days)
                return json.dumps(results, ensure_ascii=False)

            if tool_name == "email_search":
                query = args.get("query", "")
                limit = args.get("limit", 10)
                results = await integ.query(query, limit)
                return json.dumps(results, ensure_ascii=False)

            if tool_name == "email_recent":
                limit = args.get("limit", 20)
                results = await integ.get_recent(limit)
                return json.dumps(results, ensure_ascii=False)

        except Exception as exc:
            logger.exception("Integration %s tool %s failed", integration_name, tool_name)
            return f"{integration_name} 工具调用失败: {exc}"

        return f"未知 {integration_name} 工具: {tool_name}"

    def _dispatch_computer_tool(self, args: dict[str, Any]) -> str:
        """Handle computer_action tool calls."""
        try:
            from computer_use.agent import ComputerUseAgent
        except ImportError:
            return _INTEGRATION_ERRORS["computer"]

        try:
            backend = self._registry.get_backend(self._current_model)
        except ValueError:
            return "无法获取当前模型后端来执行电脑操作"

        action = args.get("action", "")
        if not action:
            return "缺少 action 参数"

        cu_agent = ComputerUseAgent(backend, verbose=False)
        try:
            result = cu_agent.run(f"执行操作: {action}，参数: {json.dumps(args, ensure_ascii=False)}")
            return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        except Exception as exc:
            return f"电脑操作失败: {exc}"
