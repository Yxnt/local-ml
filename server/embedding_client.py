"""Async client for the local Jina embedding service.

Provides:
- Single and batch embedding via HTTP (httpx)
- LRU cache with configurable size limit
- Cosine similarity helper
- Graceful error handling (connection, timeout, HTTP errors)

Usage:
    async with EmbeddingClient() as client:
        vec = await client.embed("hello world")
        vecs = await client.embed_batch(["hello", "world"])
        sim = client.cosine_similarity(vec, vecs[0])
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from collections import OrderedDict
from typing import Any

import httpx


DEFAULT_BASE_URL = "http://localhost:8001"
DEFAULT_TIMEOUT = 30.0
DEFAULT_CACHE_SIZE = 512
EMBEDDING_DIM = 768


class EmbeddingClientError(Exception):
    """Base exception for embedding client errors."""


class ConnectionError(EmbeddingClientError):
    """Cannot reach the embedding service."""


class ServiceError(EmbeddingClientError):
    """Embedding service returned an error."""


class EmbeddingClient:
    """Async client for the Jina embedding service.

    Args:
        base_url: Embedding service URL (default http://localhost:8001).
        timeout: HTTP request timeout in seconds.
        cache_size: Maximum number of cached embeddings (0 disables cache).
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        cache_size: int = DEFAULT_CACHE_SIZE,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._cache_size = cache_size
        self._client: httpx.AsyncClient | None = None

    # -- lifecycle -----------------------------------------------------------

    async def __aenter__(self) -> EmbeddingClient:
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=self._timeout,
        )
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def close(self) -> None:
        """Explicitly close the underlying HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    # -- public API ----------------------------------------------------------

    async def embed(self, text: str, task: str = "retrieval") -> list[float]:
        """Embed a single text.  Returns a 768-dim float list."""
        cache_key = self._cache_key(text, task)
        cached = self._get_cache(cache_key)
        if cached is not None:
            return cached

        result = await self._request("/embed", {"texts": [text], "task": task})
        vec = result[0]
        self._put_cache(cache_key, vec)
        return vec

    async def embed_batch(
        self,
        texts: list[str],
        task: str = "retrieval",
    ) -> list[list[float]]:
        """Embed multiple texts.  Returns a list of 768-dim float lists.

        Cache hits are served locally; cache misses are batched into a
        single HTTP request to the ``/embed/batch`` endpoint.
        """
        if not texts:
            return []

        # Partition into cache hits and misses
        results: list[list[float] | None] = [None] * len(texts)
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cache_key = self._cache_key(text, task)
            cached = self._get_cache(cache_key)
            if cached is not None:
                results[i] = cached
            else:
                miss_indices.append(i)
                miss_texts.append(text)

        # Fetch misses in one batch
        if miss_texts:
            fetched = await self._request(
                "/embed/batch", {"texts": miss_texts, "task": task}
            )
            for idx, vec in zip(miss_indices, fetched):
                results[idx] = vec
                cache_key = self._cache_key(texts[idx], task)
                self._put_cache(cache_key, vec)

        return results  # type: ignore[return-value]

    async def health(self) -> bool:
        """Return True if the embedding service is healthy."""
        client = self._ensure_client()
        try:
            resp = await client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # -- internals -----------------------------------------------------------

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError(
                "EmbeddingClient is not open. Use 'async with EmbeddingClient() as c:' "
                "or call await c.__aenter__() first."
            )
        return self._client

    async def _request(
        self,
        path: str,
        payload: dict[str, Any],
    ) -> list[list[float]]:
        """POST to the embedding service and return the embeddings list."""
        client = self._ensure_client()
        try:
            resp = await client.post(path, json=payload)
        except httpx.ConnectError as exc:
            raise ConnectionError(
                f"Cannot connect to embedding service at {self._base_url}: {exc}"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ConnectionError(
                f"Embedding service timed out after {self._timeout}s: {exc}"
            ) from exc
        except httpx.HTTPError as exc:
            raise ConnectionError(
                f"HTTP error calling embedding service: {exc}"
            ) from exc

        if resp.status_code != 200:
            raise ServiceError(
                f"Embedding service returned {resp.status_code}: {resp.text}"
            )

        data = resp.json()
        embeddings = data.get("embeddings")
        if embeddings is None:
            raise ServiceError(
                f"Unexpected response from embedding service: {data}"
            )
        return embeddings

    @staticmethod
    def _cache_key(text: str, task: str) -> str:
        return hashlib.sha256(f"{task}\x00{text}".encode()).hexdigest()

    def _get_cache(self, key: str) -> list[float] | None:
        if self._cache_size <= 0:
            return None
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _put_cache(self, key: str, vec: list[float]) -> None:
        if self._cache_size <= 0:
            return
        if key in self._cache:
            self._cache.move_to_end(key)
            return
        self._cache[key] = vec
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
