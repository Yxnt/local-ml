"""WebSocket endpoint for real-time communication.

Supports text and audio messages for voice applications.
Integrates with the ModelRegistry for inference and MemoryManager for context.

Protocol:
    Client sends:
        {
            "type": "text" | "audio",
            "content": "...",           # text or base64-encoded audio
            "model": "minicpm-v-4.6",   # optional model override
            "session_id": "abc123"      # optional session tracking
        }

    Server responds:
        {
            "type": "text" | "audio",
            "content": "...",           # text response
            "audio": "base64...",       # optional audio response
            "tool_calls": [...],        # optional tool call info
            "status": "complete" | "processing" | "error"
        }
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect

from backends.registry import ModelRegistry
from memory.manager import MemoryManager

logger = logging.getLogger(__name__)

# Default system prompt for WebSocket agent
DEFAULT_SYSTEM_PROMPT = """You are a helpful local AI assistant.
You can help users with various tasks including searching notes, managing memory, and general conversation.
Respond in the same language the user uses."""


@dataclass
class Session:
    """Tracks a WebSocket conversation session."""

    session_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    model: str | None = None


class WebSocketAgent:
    """Agent that processes messages via WebSocket.

    Wraps ModelRegistry for inference and MemoryManager for context.
    Maintains per-session conversation history.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        memory_manager: MemoryManager | None = None,
        default_model: str | None = None,
    ) -> None:
        self._registry = registry
        self._memory = memory_manager
        self._default_model = default_model or os.environ.get(
            "DEFAULT_MODEL", "gemma-4-e2b-it-4bit"
        )
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: str | None) -> Session:
        """Get existing session or create a new one."""
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        sid = session_id or str(uuid.uuid4())
        session = Session(session_id=sid)
        self._sessions[sid] = session
        return session

    def _build_system_prompt(self) -> str:
        """Build system prompt with memory context if available."""
        parts = [DEFAULT_SYSTEM_PROMPT]

        if self._memory:
            try:
                memory_prompt = self._memory.get_system_prompt()
                if memory_prompt:
                    parts.append(memory_prompt)
            except Exception as e:
                logger.warning("Failed to get memory context: %s", e)

        return "\n\n".join(parts)

    async def process_text(
        self,
        content: str,
        session: Session,
        model_override: str | None = None,
    ) -> dict[str, Any]:
        """Process a text message and return response.

        Args:
            content: User's text message.
            session: Current conversation session.
            model_override: Optional model name to use instead of default.

        Returns:
            Response dict with type, content, tool_calls, status.
        """
        model_name = model_override or session.model or self._default_model

        # Build messages for inference
        system_prompt = self._build_system_prompt()
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(session.messages)
        messages.append({"role": "user", "content": content})

        # Get backend
        try:
            backend = await self._registry.get_or_load(model_name)
        except ValueError as e:
            return {
                "type": "text",
                "content": f"Model error: {e}",
                "status": "error",
            }
        except Exception as e:
            return {
                "type": "text",
                "content": f"Model load failed: {e}",
                "status": "error",
            }

        # Build prompt and generate
        try:
            prompt = backend.apply_chat_template(
                messages,
                tools=None,
                enable_thinking=False,
            )
            output_text = backend.generate(
                prompt=prompt,
                max_tokens=2048,
                temperature=0.7,
                top_p=0.9,
            )
        except Exception as e:
            logger.error("Generation failed: %s", e)
            return {
                "type": "text",
                "content": f"Generation error: {e}",
                "status": "error",
            }

        # Parse tool calls
        tool_calls = backend.parse_tool_calls(output_text)

        # Update session history
        session.messages.append({"role": "user", "content": content})
        if tool_calls:
            session.messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": tool_calls,
            })
        else:
            session.messages.append({
                "role": "assistant",
                "content": output_text,
            })

        # Build response
        response: dict[str, Any] = {
            "type": "text",
            "status": "complete",
        }

        if tool_calls:
            response["content"] = output_text
            response["tool_calls"] = tool_calls
        else:
            response["content"] = output_text

        return response

    async def process_audio(
        self,
        content: str,
        session: Session,
    ) -> dict[str, Any]:
        """Process an audio message (placeholder for future STT/TTS).

        Args:
            content: Base64-encoded audio data.
            session: Current conversation session.

        Returns:
            Response dict acknowledging audio receipt.
        """
        # Placeholder: real STT/TTS will be added later
        return {
            "type": "text",
            "content": "Audio received. Speech-to-text and text-to-speech support will be added in a future update.",
            "status": "complete",
        }

    def clear_session(self, session_id: str) -> None:
        """Remove a session."""
        self._sessions.pop(session_id, None)


def handle_websocket(
    app_or_router: Any,
    registry: ModelRegistry,
    memory_manager: MemoryManager | None = None,
    path: str = "/ws",
) -> None:
    """Register WebSocket endpoint on a FastAPI app or router.

    Args:
        app_or_router: FastAPI app or APIRouter instance.
        registry: ModelRegistry for model access.
        memory_manager: Optional MemoryManager for context.
        path: WebSocket endpoint path (default /ws).
    """
    agent = WebSocketAgent(
        registry=registry,
        memory_manager=memory_manager,
    )

    @app_or_router.websocket(path)
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        logger.info("WebSocket client connected")

        # Track session for this connection (maintains continuity if no session_id provided)
        current_session: Session | None = None

        try:
            while True:
                # Receive message
                raw = await websocket.receive_text()

                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    await websocket.send_json({
                        "type": "text",
                        "content": "Invalid JSON",
                        "status": "error",
                    })
                    continue

                msg_type = data.get("type", "text")
                content = data.get("content", "")
                model_override = data.get("model")
                session_id = data.get("session_id")

                # Get or create session
                # If no session_id provided, reuse the current session for this connection
                if session_id:
                    current_session = agent.get_or_create_session(session_id)
                elif current_session is None:
                    current_session = agent.get_or_create_session(None)

                # Send processing status
                await websocket.send_json({"status": "processing"})

                # Process based on type
                if msg_type == "audio":
                    response = await agent.process_audio(content, current_session)
                else:
                    response = await agent.process_text(
                        content=content,
                        session=current_session,
                        model_override=model_override,
                    )

                # Include session_id in response
                response["session_id"] = current_session.session_id

                # Send response
                await websocket.send_json(response)

        except WebSocketDisconnect:
            logger.info("WebSocket client disconnected")
        except Exception as e:
            logger.error("WebSocket error: %s", e)
            try:
                await websocket.send_json({
                    "type": "text",
                    "content": f"Server error: {e}",
                    "status": "error",
                })
            except Exception:
                pass
