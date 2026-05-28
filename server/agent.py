"""Core Agent - ties together model, memory, integrations, and context.

Usage:
    agent = Agent()
    response = await agent.run("Hello!")
    await agent.close()
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from backends.registry import ModelRegistry
from memory.manager import MemoryManager
from optimization.collector import UsageCollector, Outcome
from server.context_manager import ContextManager
from server.embedding_client import EmbeddingClient

logger = logging.getLogger(__name__)

# -- Tool dispatch table --------------------------------------------------
# Maps tool name prefixes to the component that handles them.
_MEMORY_TOOLS = {"memory_remember", "memory_recall", "memory_stats"}
_OBSIDIAN_TOOLS = {"obsidian_search", "obsidian_read"}
_CALENDAR_TOOLS = {"calendar_search", "calendar_upcoming"}
_EMAIL_TOOLS = {"email_search", "email_recent"}
_COMPUTER_TOOLS = {"computer_action"}

# Friendly error messages for unconfigured integrations.
_INTEGRATION_ERRORS: dict[str, str] = {
    "obsidian": "请先配置 Obsidian 路径（在 config.yaml 的 integrations.obsidian.vaults 中设置）",
    "calendar": "请先配置日历账号（在 config.yaml 的 integrations.calendar.accounts 中设置）",
    "email": "请先配置邮箱账号（在 config.yaml 的 integrations.email.accounts 中设置）",
    "computer": "请先配置电脑控制功能（computer_use 模块）",
}


class Agent:
    """High-level agent that orchestrates model, memory, tools, and integrations.

    Components are initialised lazily where possible so the agent can start
    without every integration being configured.

    Args:
        data_dir: Directory for memory data files (soul, user, db).
        default_model: Model name to use when none is specified.
        max_context_turns: Max conversation turns to keep in context window.
    """

    def __init__(
        self,
        data_dir: str = "memory/data",
        default_model: str = "gemma-4-e2b-it-4bit",
        max_context_turns: int = 20,
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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Persist state and release resources."""
        self._collector.end_session()
        self._collector.disconnect()
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
        max_rounds = 5  # prevent infinite tool loops

        for _ in range(max_rounds):
            tools = self.get_tools()
            prompt = backend.apply_chat_template(
                self._messages,
                tools=tools if tools else None,
            )
            output_text = backend.generate(prompt=prompt)

            tool_calls = backend.parse_tool_calls(output_text)

            if not tool_calls:
                # Plain text response -- done.
                self._messages.append({"role": "assistant", "content": output_text})

                # Update context manager.
                self._context.update(user_input, output_text)

                # Record interaction.
                self._collector.record_interaction(
                    user_input=user_input,
                    agent_response=output_text,
                    tool_calls=all_tool_calls or None,
                    tool_results=all_tool_results or None,
                    outcome=Outcome.SUCCESS if all_tool_calls else Outcome.UNKNOWN,
                    metadata={"model": model_name, "session_id": self._session_id},
                )
                return output_text

            # Execute tool calls.
            self._messages.append({
                "role": "assistant",
                "content": output_text,
                "tool_calls": tool_calls,
            })
            all_tool_calls.extend(tool_calls)

            tool_results = await self.execute_tools(tool_calls)
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
        return fallback

    async def execute_tools(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute tool calls and return results.

        Each result dict has keys: ``tool_call_id``, ``name``, ``content``.
        Errors are caught and returned as content strings rather than raised.
        """
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
                content = await self._dispatch_tool(name, args)
            except Exception as exc:
                logger.exception("Tool %s raised an error", name)
                content = f"工具调用失败: {exc}"

            results.append({
                "tool_call_id": call_id,
                "name": name,
                "content": content if isinstance(content, str) else json.dumps(content, ensure_ascii=False),
            })

        return results

    def get_tools(self) -> list[dict[str, Any]]:
        """Collect tool definitions from memory and connected integrations."""
        tools: list[dict[str, Any]] = []

        # Memory tools are always available.
        tools.extend(self._memory.get_tools())

        # Integration tools -- only from successfully connected integrations.
        for name, integ in self._integrations.items():
            if hasattr(integ, "get_tools"):
                try:
                    tools.extend(integ.get_tools())
                except Exception:
                    logger.warning("Failed to get tools from %s", name)

        # Computer use tool (always expose it; error handled at dispatch).
        if "computer" not in self._integration_errors:
            from computer_use.tools import COMPUTER_USE_TOOL
            tools.append(COMPUTER_USE_TOOL)

        return tools

    async def switch_model(self, model_name: str) -> None:
        """Switch to a different model backend.

        The currently loaded model is unloaded first (hot-switch via registry).
        """
        # Validate the model exists in the registry.
        self._registry.get_backend(model_name)  # raises ValueError if unknown
        # Pre-load (triggers unload of previous).
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

        Returns True on success, False on failure (error stored internally).
        """
        try:
            if hasattr(integration, "connect"):
                await integration.connect()
            self._integrations[name] = integration
            self._integration_errors.pop(name, None)
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self) -> str:
        """Assemble system prompt from soul + user + memory + context."""
        parts: list[str] = []

        # Soul + user + recent memories (from MemoryManager).
        memory_prompt = self._memory.get_system_prompt()
        if memory_prompt:
            parts.append(memory_prompt)

        # Contextual entities and recent history.
        context_info = self._context.get_relevant_context("")
        if context_info:
            parts.append(f"\n会话上下文：\n{context_info}")

        # Available integration summary.
        connected = [n for n in self._integrations if n not in self._integration_errors]
        if connected:
            parts.append(f"\n已连接的数据源：{', '.join(connected)}")

        return "\n\n".join(parts)

    async def _dispatch_tool(self, name: str, args: dict[str, Any]) -> str:
        """Route a tool call to the correct handler and return the result."""

        # -- Memory tools -------------------------------------------------
        if name in _MEMORY_TOOLS:
            return self._dispatch_memory_tool(name, args)

        # -- Obsidian tools -----------------------------------------------
        if name in _OBSIDIAN_TOOLS:
            return await self._dispatch_integration_tool(
                "obsidian", name, args,
                error_hint=_INTEGRATION_ERRORS["obsidian"],
            )

        # -- Calendar tools -----------------------------------------------
        if name in _CALENDAR_TOOLS:
            return await self._dispatch_integration_tool(
                "calendar", name, args,
                error_hint=_INTEGRATION_ERRORS["calendar"],
            )

        # -- Email tools --------------------------------------------------
        if name in _EMAIL_TOOLS:
            return await self._dispatch_integration_tool(
                "email", name, args,
                error_hint=_INTEGRATION_ERRORS["email"],
            )

        # -- Computer use tools -------------------------------------------
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

        # ComputerUseAgent requires a loaded backend.
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
