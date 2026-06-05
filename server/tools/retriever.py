"""ToolRetriever — semantic search over the ToolRegistry.

Uses Jina embeddings via the local embedding service to find the most relevant
tools for a given user task.  Falls back to keyword matching when the embedding
service is unavailable.
"""

from __future__ import annotations

import logging
from typing import Any

from server.embedding_client import EmbeddingClient
from server.tools.registry import ToolRegistry
from server.tools.spec import ToolSpec, ToolStatus

logger = logging.getLogger(__name__)


class ToolRetriever:
    """Retrieve the top-k most relevant tools for a user query.

    Two strategies are combined:
    1. **Semantic**: embed the query and search ``tools_vec`` via sqlite-vec.
    2. **Keyword**: LIKE-match against tool names and descriptions.

    Results are merged and deduplicated, favouring semantic matches.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        embedding_client: EmbeddingClient | None = None,
    ) -> None:
        self._registry = registry
        self._embedding = embedding_client

    async def retrieve(self, query: str, limit: int = 10) -> list[ToolSpec]:
        """Return the most relevant active tools for *query*."""
        results: list[ToolSpec] = []
        seen: set[str] = set()

        # 1. Semantic search (best-effort).
        if self._embedding is not None:
            try:
                if await self._embedding.health():
                    q_vec = await self._embedding.embed(query, task="retrieval")
                    vec_hits = self._registry.search_by_embedding(q_vec, limit=limit)
                    for spec, dist in vec_hits:
                        if spec.name not in seen and spec.status == ToolStatus.ACTIVE:
                            results.append(spec)
                            seen.add(spec.name)
            except Exception as e:
                logger.debug("Semantic tool search failed, falling back to keyword: %s", e)

        # 2. Keyword fallback / supplement.
        if len(results) < limit:
            keyword_hits = self._keyword_search(query, limit=limit * 2)
            for spec in keyword_hits:
                if spec.name not in seen and spec.status == ToolStatus.ACTIVE:
                    results.append(spec)
                    seen.add(spec.name)
                if len(results) >= limit:
                    break

        return results

    def _keyword_search(self, query: str, limit: int = 20) -> list[ToolSpec]:
        """Word-level match against tool names, descriptions, and tags."""
        words = query.lower().split()
        all_tools = self._registry.list_tools(status=ToolStatus.ACTIVE)

        scored: list[tuple[int, ToolSpec]] = []
        for spec in all_tools:
            score = 0
            name_lower = spec.name.lower()
            desc_lower = spec.description.lower()
            tags_lower = [t.lower() for t in spec.tags]
            provider_lower = spec.provider.lower()

            for word in words:
                if word in name_lower:
                    score += 10
                if word in desc_lower:
                    score += 5
                if any(word in tag for tag in tags_lower):
                    score += 3
                if word in provider_lower:
                    score += 2

            if score > 0:
                scored.append((score, spec))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [spec for _, spec in scored[:limit]]

    async def embed_all_tools(self) -> int:
        """Batch-embed all tools that don't yet have an embedding.

        Returns the number of tools newly embedded.
        """
        if self._embedding is None:
            return 0

        try:
            if not await self._embedding.health():
                logger.warning("Embedding service not available")
                return 0
        except Exception:
            return 0

        all_tools = self._registry.list_tools(status=None)
        to_embed: list[tuple[str, str]] = []  # (name, text)
        for spec in all_tools:
            if spec.embedding is None:
                text = f"{spec.name}: {spec.description}"
                to_embed.append((spec.name, text))

        if not to_embed:
            return 0

        texts = [t for _, t in to_embed]
        names = [n for n, _ in to_embed]

        try:
            embeddings = await self._embedding.embed_batch(texts, task="retrieval")
        except Exception as e:
            logger.error("Batch embedding failed: %s", e)
            return 0

        count = 0
        for name, emb in zip(names, embeddings):
            self._registry.set_embedding(name, emb)
            count += 1

        logger.info("Embedded %d tools", count)
        return count
