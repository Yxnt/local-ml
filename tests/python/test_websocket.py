"""Tests for WebSocket endpoint."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from backends.registry import ModelRegistry
from server.websocket import WebSocketAgent, Session, handle_websocket


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_backend():
    """Create a mock model backend."""
    backend = MagicMock()
    backend.apply_chat_template.return_value = "test prompt"
    backend.generate.return_value = "Hello! How can I help you?"
    backend.parse_tool_calls.return_value = []
    return backend


@pytest.fixture
def mock_registry(mock_backend):
    """Create a mock registry with a loaded backend."""
    registry = MagicMock(spec=ModelRegistry)
    registry.get_or_load = AsyncMock(return_value=mock_backend)
    return registry


@pytest.fixture
def agent(mock_registry):
    """Create a WebSocketAgent with mocked registry."""
    return WebSocketAgent(registry=mock_registry)


@pytest.fixture
def app_with_ws(mock_registry):
    """Create a FastAPI app with WebSocket endpoint."""
    app = FastAPI()
    handle_websocket(app, mock_registry)
    return app


# ---------------------------------------------------------------------------
# Session tests
# ---------------------------------------------------------------------------


class TestSession:
    def test_session_creation(self):
        session = Session(session_id="test-123")
        assert session.session_id == "test-123"
        assert session.messages == []
        assert session.model is None

    def test_session_with_model(self):
        session = Session(session_id="test-123", model="minicpm-v-4.6")
        assert session.model == "minicpm-v-4.6"


# ---------------------------------------------------------------------------
# WebSocketAgent tests
# ---------------------------------------------------------------------------


class TestWebSocketAgent:
    def test_get_or_create_session_new(self, agent):
        session = agent.get_or_create_session(None)
        assert session.session_id is not None
        assert len(session.session_id) > 0
        assert session.messages == []

    def test_get_or_create_session_with_id(self, agent):
        session = agent.get_or_create_session("my-session")
        assert session.session_id == "my-session"
        # Should return same session on second call
        session2 = agent.get_or_create_session("my-session")
        assert session2 is session

    def test_clear_session(self, agent):
        agent.get_or_create_session("to-clear")
        assert "to-clear" in agent._sessions
        agent.clear_session("to-clear")
        assert "to-clear" not in agent._sessions

    @pytest.mark.asyncio
    async def test_process_text(self, agent, mock_backend):
        session = agent.get_or_create_session("test-session")
        response = await agent.process_text("Hello", session)

        assert response["type"] == "text"
        assert response["status"] == "complete"
        assert response["content"] == "Hello! How can I help you?"
        assert "tool_calls" not in response

        # Check session history updated
        assert len(session.messages) == 2
        assert session.messages[0] == {"role": "user", "content": "Hello"}
        assert session.messages[1] == {
            "role": "assistant",
            "content": "Hello! How can I help you?",
        }

    @pytest.mark.asyncio
    async def test_process_text_with_model_override(self, agent, mock_registry):
        session = agent.get_or_create_session("test-session")
        await agent.process_text("Hello", session, model_override="minicpm-v-4.6")

        # Should call with overridden model
        mock_registry.get_or_load.assert_called_with("minicpm-v-4.6")

    @pytest.mark.asyncio
    async def test_process_text_with_tool_calls(self, agent, mock_backend):
        mock_backend.parse_tool_calls.return_value = [
            {
                "id": "call_0",
                "type": "function",
                "function": {
                    "name": "search_notes",
                    "arguments": '{"query": "test"}',
                },
            }
        ]
        mock_backend.generate.return_value = '<|tool_call>call:search_notes{query:"test"}<tool_call|>'

        session = agent.get_or_create_session("test-session")
        response = await agent.process_text("Search for test", session)

        assert response["status"] == "complete"
        assert "tool_calls" in response
        assert len(response["tool_calls"]) == 1
        assert response["tool_calls"][0]["function"]["name"] == "search_notes"

    @pytest.mark.asyncio
    async def test_process_text_model_error(self, mock_registry):
        mock_registry.get_or_load = AsyncMock(
            side_effect=ValueError("Unknown model: bad-model")
        )
        agent = WebSocketAgent(registry=mock_registry)

        session = agent.get_or_create_session("test-session")
        response = await agent.process_text("Hello", session, model_override="bad-model")

        assert response["status"] == "error"
        assert "Unknown model" in response["content"]

    @pytest.mark.asyncio
    async def test_process_text_generation_error(self, agent, mock_backend):
        mock_backend.generate.side_effect = RuntimeError("CUDA out of memory")

        session = agent.get_or_create_session("test-session")
        response = await agent.process_text("Hello", session)

        assert response["status"] == "error"
        assert "Generation error" in response["content"]

    @pytest.mark.asyncio
    async def test_process_audio_placeholder(self, agent):
        session = agent.get_or_create_session("test-session")
        response = await agent.process_audio("base64audiodata", session)

        assert response["type"] == "text"
        assert response["status"] == "complete"
        assert "Audio received" in response["content"]

    def test_multi_turn_history(self, agent, mock_backend):
        """Test that conversation history accumulates across turns."""
        import asyncio

        session = agent.get_or_create_session("test-session")

        # Simulate two turns
        asyncio.run(agent.process_text("Hello", session))
        asyncio.run(agent.process_text("How are you?", session))

        # Should have 4 messages: user1, assistant1, user2, assistant2
        assert len(session.messages) == 4
        assert session.messages[0]["content"] == "Hello"
        assert session.messages[2]["content"] == "How are you?"


# ---------------------------------------------------------------------------
# WebSocket endpoint integration tests
# ---------------------------------------------------------------------------


class TestWebSocketEndpoint:
    def test_websocket_connection(self, app_with_ws):
        """Test that WebSocket endpoint accepts connections."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            # Connection should succeed
            assert ws is not None

    def test_text_message_exchange(self, app_with_ws, mock_backend):
        """Test sending and receiving text messages."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            # Send text message
            ws.send_json({
                "type": "text",
                "content": "Hello",
            })

            # Receive processing status
            response = ws.receive_json()
            assert response["status"] == "processing"

            # Receive actual response
            response = ws.receive_json()
            assert response["type"] == "text"
            assert response["status"] == "complete"
            assert response["content"] == "Hello! How can I help you?"
            assert "session_id" in response

    def test_session_tracking(self, app_with_ws, mock_backend):
        """Test that session ID is tracked across messages."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            # Send with session ID
            ws.send_json({
                "type": "text",
                "content": "Hello",
                "session_id": "my-session-123",
            })

            # Skip processing status
            ws.receive_json()

            # Get response
            response = ws.receive_json()
            assert response["session_id"] == "my-session-123"

    def test_model_override(self, app_with_ws, mock_backend):
        """Test model override in WebSocket message."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "text",
                "content": "Hello",
                "model": "minicpm-v-4.6",
            })

            # Skip processing status
            ws.receive_json()

            # Get response
            response = ws.receive_json()
            assert response["status"] == "complete"

    def test_audio_message(self, app_with_ws):
        """Test audio message handling (placeholder)."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({
                "type": "audio",
                "content": "base64audiodata",
            })

            # Skip processing status
            ws.receive_json()

            # Get response
            response = ws.receive_json()
            assert response["type"] == "text"
            assert "Audio received" in response["content"]

    def test_invalid_json(self, app_with_ws):
        """Test handling of invalid JSON."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            ws.send_text("not valid json {{{")

            response = ws.receive_json()
            assert response["status"] == "error"
            assert "Invalid JSON" in response["content"]

    def test_multiple_messages(self, app_with_ws, mock_backend):
        """Test multiple messages on same connection."""
        client = TestClient(app_with_ws)
        with client.websocket_connect("/ws") as ws:
            # First message
            ws.send_json({"type": "text", "content": "Hello"})
            ws.receive_json()  # processing
            response1 = ws.receive_json()
            assert response1["status"] == "complete"

            # Second message
            ws.send_json({"type": "text", "content": "How are you?"})
            ws.receive_json()  # processing
            response2 = ws.receive_json()
            assert response2["status"] == "complete"

            # Both should have same session (auto-generated)
            assert response1["session_id"] == response2["session_id"]


# ---------------------------------------------------------------------------
# Error handling tests
# ---------------------------------------------------------------------------


class TestWebSocketErrorHandling:
    def test_model_load_error(self):
        """Test error response when model fails to load."""
        registry = MagicMock(spec=ModelRegistry)
        registry.get_or_load = AsyncMock(
            side_effect=Exception("GPU memory exhausted")
        )

        app = FastAPI()
        handle_websocket(app, registry)

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "text", "content": "Hello"})
            ws.receive_json()  # processing
            response = ws.receive_json()

            assert response["status"] == "error"
            assert "Model load failed" in response["content"]

    def test_generation_error(self):
        """Test error response when generation fails."""
        backend = MagicMock()
        backend.apply_chat_template.return_value = "prompt"
        backend.generate.side_effect = RuntimeError("OOM")

        registry = MagicMock(spec=ModelRegistry)
        registry.get_or_load = AsyncMock(return_value=backend)

        app = FastAPI()
        handle_websocket(app, registry)

        client = TestClient(app)
        with client.websocket_connect("/ws") as ws:
            ws.send_json({"type": "text", "content": "Hello"})
            ws.receive_json()  # processing
            response = ws.receive_json()

            assert response["status"] == "error"
            assert "Generation error" in response["content"]
