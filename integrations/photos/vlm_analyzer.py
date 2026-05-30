"""VLM-based image analyzer for Apple Photos with result caching.

Provides vision-language model analysis of photos, caching results in
the same SQLite database used by PhotosIndexer.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from backends.registry import ModelRegistry

logger = logging.getLogger(__name__)


class VLMAnalyzer:
    """Analyze photos using a vision-language model with result caching.

    Parameters
    ----------
    model_name:
        Name of the model in ModelRegistry (e.g. ``"minicpm-v-4.6"``).
    db_path:
        Path to the SQLite database (shared with PhotosIndexer).
    """

    def __init__(self, model_name: str, db_path: str) -> None:
        self.model_name = model_name
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None
        self._registry: ModelRegistry | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Connect to the existing SQLite database and ensure the
        ``photo_analyses`` table exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS photos (
                uuid              TEXT PRIMARY KEY,
                original_filename TEXT
            );

            CREATE TABLE IF NOT EXISTS photo_analyses (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_uuid TEXT NOT NULL REFERENCES photos(uuid),
                model      TEXT,
                analysis   TEXT,
                created_at TEXT,
                UNIQUE(photo_uuid, model)
            );
            """
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # analysis
    # ------------------------------------------------------------------

    def analyze(self, photo_path: str, question: str) -> str:
        """Analyze a photo with the VLM, returning cached result if available.

        The model is lazy-loaded on first call via :class:`ModelRegistry`.

        Parameters
        ----------
        photo_path:
            Filesystem path to the image file.
        question:
            Natural-language question about the image.

        Returns
        -------
        str
            The model's analysis text.
        """
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        photo_id = self._photo_id(photo_path)

        # Check cache first
        cached = self._get_cached(photo_id, question)
        if cached is not None:
            return cached

        # Lazy-load model
        if self._registry is None:
            self._registry = ModelRegistry()
            self._registry.register_defaults()

        backend = self._registry.get_backend(self.model_name)

        # Load model if not already active
        if not getattr(backend, "_model", None):
            import asyncio
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(self._registry.get_or_load(self.model_name))
            finally:
                loop.close()

        # Run VLM inference with image
        from mlx_vlm import generate as vlm_generate

        output = vlm_generate(
            model=backend._model,
            processor=backend._processor,
            prompt=question,
            image=photo_path,
            max_tokens=1024,
            temperature=0.3,
            verbose=False,
        )
        analysis = getattr(output, "text", str(output))

        # Cache the result
        self._cache_result(photo_id, question, analysis)
        return analysis

    # ------------------------------------------------------------------
    # cache helpers
    # ------------------------------------------------------------------

    def _get_cached(self, photo_id: str, question: str) -> str | None:
        """Return cached analysis for *photo_id* and *question*, or ``None``."""
        if not self._conn:
            return None
        cache_key = self._cache_key(question)
        row = self._conn.execute(
            "SELECT analysis FROM photo_analyses WHERE photo_uuid = ? AND model = ?",
            (photo_id, cache_key),
        ).fetchone()
        return row["analysis"] if row else None

    def _cache_result(self, photo_id: str, question: str, analysis: str) -> None:
        """Store an analysis result in the cache."""
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")
        cache_key = self._cache_key(question)
        self._conn.execute(
            """
            INSERT OR REPLACE INTO photo_analyses
                (photo_uuid, model, analysis, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (photo_id, cache_key, analysis, datetime.now(tz=None).isoformat()),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(question: str) -> str:
        """Build a composite cache key encoding the question."""
        return f"vlm:{question}"

    @staticmethod
    def _photo_id(photo_path: str) -> str:
        """Derive a stable photo identifier from its filesystem path."""
        return Path(photo_path).stem
