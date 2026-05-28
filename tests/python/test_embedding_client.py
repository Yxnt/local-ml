"""Tests for server.embedding_client -- async embedding client with cache."""

import math
import pytest
import httpx

from server.embedding_client import (
    DEFAULT_BASE_URL,
    EmbeddingClient,
    ConnectionError as EmbedConnectionError,
    ServiceError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_embedding(seed: int = 0) -> list[float]:
    """Return a deterministic 768-dim unit vector."""
    vec = [float((seed + i) % 10) for i in range(768)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


class MockTransport(httpx.AsyncBaseTransport):
    """In-memory transport that mimics the embedding service."""

    def __init__(self):
        self.call_log: list[tuple[str, dict]] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        body = request.read()
        import json
        payload = json.loads(body) if body else {}

        self.call_log.append((url, payload))

        if "/health" in url:
            return httpx.Response(200, json={"status": "ok"})

        texts = payload.get("texts", [])
        embeddings = [_fake_embedding(hash(t) % 100) for t in texts]
        return httpx.Response(200, json={"embeddings": embeddings})


class ErrorTransport(httpx.AsyncBaseTransport):
    """Transport that always raises ConnectError."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")


class TimeoutTransport(httpx.AsyncBaseTransport):
    """Transport that always raises TimeoutException."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise httpx.TimeoutException("Read timed out")


class HttpErrorTransport(httpx.AsyncBaseTransport):
    """Transport that returns 500."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="internal server error")


class MalformedTransport(httpx.AsyncBaseTransport):
    """Transport that returns 200 but missing 'embeddings' key."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"wrong_key": []})


def _make_client(transport=None, cache_size=512) -> EmbeddingClient:
    """Create an EmbeddingClient with a mock transport (no real server)."""
    client = EmbeddingClient(base_url=DEFAULT_BASE_URL, cache_size=cache_size)
    client._client = httpx.AsyncClient(
        base_url=DEFAULT_BASE_URL,
        transport=transport or MockTransport(),
    )
    return client


# ---------------------------------------------------------------------------
# Tests: single embedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_single():
    transport = MockTransport()
    client = _make_client(transport)

    vec = await client.embed("hello world")

    assert isinstance(vec, list)
    assert len(vec) == 768
    assert all(isinstance(x, float) for x in vec)
    # Verify request was sent to /embed
    assert any("/embed" in url and "/batch" not in url for url, _ in transport.call_log)

    await client.close()


@pytest.mark.asyncio
async def test_embed_returns_correct_values():
    client = _make_client()
    vec = await client.embed("test text")
    expected = _fake_embedding(hash("test text") % 100)
    assert vec == expected
    await client.close()


# ---------------------------------------------------------------------------
# Tests: batch embedding
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_embed_batch():
    transport = MockTransport()
    client = _make_client(transport)

    texts = ["hello", "world", "foo"]
    vecs = await client.embed_batch(texts)

    assert len(vecs) == 3
    for vec in vecs:
        assert isinstance(vec, list)
        assert len(vec) == 768

    # Verify request was sent to /embed/batch
    assert any("/embed/batch" in url for url, _ in transport.call_log)

    await client.close()


@pytest.mark.asyncio
async def test_embed_batch_empty():
    client = _make_client()
    result = await client.embed_batch([])
    assert result == []
    await client.close()


@pytest.mark.asyncio
async def test_embed_batch_with_cache_partial_hits():
    """Pre-populate cache for one text, verify only misses are fetched."""
    transport = MockTransport()
    client = _make_client(transport)

    # Prime the cache for "hello"
    await client.embed("hello")
    transport.call_log.clear()

    # Now batch with "hello" (cached) + "world" (miss)
    vecs = await client.embed_batch(["hello", "world"])

    assert len(vecs) == 2
    # Only one HTTP call should have been made (for the miss)
    batch_calls = [c for c in transport.call_log if "/embed/batch" in c[0]]
    assert len(batch_calls) == 1
    # The batch call should only contain the miss text
    assert batch_calls[0][1]["texts"] == ["world"]

    await client.close()


# ---------------------------------------------------------------------------
# Tests: cache
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit():
    transport = MockTransport()
    client = _make_client(transport)

    vec1 = await client.embed("cached text")
    call_count_after_first = len(transport.call_log)

    vec2 = await client.embed("cached text")

    # No additional HTTP call
    assert len(transport.call_log) == call_count_after_first
    # Same result
    assert vec1 == vec2

    await client.close()


@pytest.mark.asyncio
async def test_cache_size_limit():
    transport = MockTransport()
    client = _make_client(transport, cache_size=2)

    await client.embed("a")
    await client.embed("b")
    await client.embed("c")  # evicts "a"

    transport.call_log.clear()

    # "a" was evicted, should trigger a new request
    await client.embed("a")
    assert len(transport.call_log) == 1

    # "c" is still cached
    transport.call_log.clear()
    await client.embed("c")
    assert len(transport.call_log) == 0

    await client.close()


@pytest.mark.asyncio
async def test_cache_disabled():
    transport = MockTransport()
    client = _make_client(transport, cache_size=0)

    await client.embed("no cache")
    transport.call_log.clear()

    await client.embed("no cache")
    # Should have made another request
    assert len(transport.call_log) == 1

    await client.close()


@pytest.mark.asyncio
async def test_cache_key_differentiates_task():
    transport = MockTransport()
    client = _make_client(transport)

    await client.embed("same text", task="retrieval")
    transport.call_log.clear()

    # Different task -> different cache key
    await client.embed("same text", task="clustering")
    assert len(transport.call_log) == 1

    await client.close()


# ---------------------------------------------------------------------------
# Tests: cosine similarity
# ---------------------------------------------------------------------------


def test_cosine_similarity_identical():
    vec = [1.0, 0.0, 0.0]
    assert EmbeddingClient.cosine_similarity(vec, vec) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert EmbeddingClient.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite():
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert EmbeddingClient.cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    a = [0.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert EmbeddingClient.cosine_similarity(a, b) == 0.0


def test_cosine_similarity_arbitrary():
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    dot = 1 * 4 + 2 * 5 + 3 * 6  # 32
    norm_a = math.sqrt(1 + 4 + 9)
    norm_b = math.sqrt(16 + 25 + 36)
    expected = dot / (norm_a * norm_b)
    assert EmbeddingClient.cosine_similarity(a, b) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_connection_error():
    client = _make_client(ErrorTransport())

    with pytest.raises(EmbedConnectionError, match="Cannot connect"):
        await client.embed("hello")

    await client.close()


@pytest.mark.asyncio
async def test_timeout_error():
    client = _make_client(TimeoutTransport())

    with pytest.raises(EmbedConnectionError, match="timed out"):
        await client.embed("hello")

    await client.close()


@pytest.mark.asyncio
async def test_http_error():
    client = _make_client(HttpErrorTransport())

    with pytest.raises(ServiceError, match="500"):
        await client.embed("hello")

    await client.close()


@pytest.mark.asyncio
async def test_malformed_response():
    client = _make_client(MalformedTransport())

    with pytest.raises(ServiceError, match="Unexpected response"):
        await client.embed("hello")

    await client.close()


@pytest.mark.asyncio
async def test_batch_connection_error():
    client = _make_client(ErrorTransport())

    with pytest.raises(EmbedConnectionError):
        await client.embed_batch(["a", "b"])

    await client.close()


@pytest.mark.asyncio
async def test_batch_service_error():
    client = _make_client(HttpErrorTransport())

    with pytest.raises(ServiceError):
        await client.embed_batch(["a", "b"])

    await client.close()


@pytest.mark.asyncio
async def test_not_open_error():
    """Using the client without async-with raises RuntimeError."""
    client = EmbeddingClient()
    with pytest.raises(RuntimeError, match="not open"):
        await client.embed("hello")


# ---------------------------------------------------------------------------
# Tests: health check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_health_ok():
    client = _make_client()
    assert await client.health() is True
    await client.close()


@pytest.mark.asyncio
async def test_health_unreachable():
    client = _make_client(ErrorTransport())
    assert await client.health() is False
    await client.close()


# ---------------------------------------------------------------------------
# Tests: context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_context_manager():
    transport = MockTransport()
    async with EmbeddingClient() as client:
        # Replace the real httpx client with one using mock transport
        await client._client.aclose()
        client._client = httpx.AsyncClient(
            base_url=DEFAULT_BASE_URL,
            transport=transport,
        )
        vec = await client.embed("context test")
        assert len(vec) == 768

    # After exiting, client._client should be None
    assert client._client is None
