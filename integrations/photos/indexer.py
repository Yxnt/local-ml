"""Apple Photos metadata indexer.

Syncs photo metadata from the macOS Photos library (via osxphotos) into a
local SQLite cache for fast searching without hitting the Photos database on
every query.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any


class PhotosIndexer:
    """Index Apple Photos metadata into a local SQLite cache.

    Parameters
    ----------
    library_path:
        Path to the Photos library (e.g. ``~/Pictures/Photos Library.photoslibrary``).
    db_path:
        Where to store the SQLite cache database.
    """

    def __init__(self, library_path: str, db_path: str) -> None:
        self.library_path = Path(library_path).expanduser()
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def init_db(self) -> None:
        """Create the SQLite tables if they don't already exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS photos (
                uuid             TEXT PRIMARY KEY,
                original_filename TEXT,
                date             TEXT,       -- ISO-8601
                width            INTEGER,
                height           INTEGER,
                latitude         REAL,
                longitude        REAL,
                location_name    TEXT,
                albums           TEXT,       -- JSON array
                tags             TEXT,       -- JSON array
                description      TEXT,
                media_type       TEXT,
                file_size        INTEGER,
                local_id         TEXT,       -- osxphotos local identifier
                updated_at       TEXT
            );

            CREATE TABLE IF NOT EXISTS photo_analyses (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                photo_uuid TEXT NOT NULL REFERENCES photos(uuid),
                model     TEXT,
                analysis  TEXT,
                created_at TEXT,
                UNIQUE(photo_uuid, model)
            );

            CREATE INDEX IF NOT EXISTS idx_photos_date ON photos(date);
            CREATE INDEX IF NOT EXISTS idx_photos_location ON photos(latitude, longitude);
            """
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # sync
    # ------------------------------------------------------------------

    def sync_sync(self) -> dict[str, int]:
        """Incremental sync from the Photos library using osxphotos.

        Returns a dict with ``new``, ``updated``, and ``total`` counts.
        """
        try:
            import osxphotos  # noqa: F811
        except ImportError:
            raise ImportError(
                "osxphotos is required for sync: pip install osxphotos"
            )

        photosdb = osxphotos.PhotosDB(str(self.library_path))
        new_count = 0
        updated_count = 0

        for photo in photosdb.photos():
            metadata = self._extract_metadata(photo)
            cached = self._get_cached(metadata["uuid"])
            if cached is None:
                self._insert(metadata)
                new_count += 1
            else:
                self._update(metadata)
                updated_count += 1

        if self._conn:
            self._conn.commit()

        total = self._count_photos()
        return {"new": new_count, "updated": updated_count, "total": total}

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def search_sync(
        self,
        query: str | None = None,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        album: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Search cached photo metadata.

        Parameters
        ----------
        query:
            Free-text search across location_name, tags, description, and
            original_filename.
        date_from / date_to:
            Inclusive date range filter.
        album:
            Filter to photos belonging to this album.
        limit:
            Maximum number of results.
        """
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        conditions: list[str] = []
        params: list[Any] = []

        if query:
            conditions.append(
                "("
                "location_name LIKE ? OR "
                "tags LIKE ? OR "
                "description LIKE ? OR "
                "original_filename LIKE ?"
                ")"
            )
            like = f"%{query}%"
            params.extend([like, like, like, like])

        if date_from is not None:
            conditions.append("date >= ?")
            params.append(date_from.isoformat())

        if date_to is not None:
            conditions.append("date <= ?")
            params.append(date_to.isoformat())

        if album:
            conditions.append("albums LIKE ?")
            params.append(f"%{album}%")

        where = " AND ".join(conditions) if conditions else "1=1"
        sql = f"SELECT * FROM photos WHERE {where} ORDER BY date DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def get_photo_path(self, uuid: str) -> str | None:
        """Return the original file path for a photo, or ``None``."""
        if not self._conn:
            return None
        try:
            import osxphotos  # noqa: F811

            photosdb = osxphotos.PhotosDB(str(self.library_path))
            for p in photosdb.photos(uuid=[uuid]):
                if p.original_filename:
                    return str(p.path)
        except ImportError:
            pass
        return None

    def list_albums_sync(self) -> list[str]:
        """Return a sorted list of unique album names."""
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        rows = self._conn.execute("SELECT DISTINCT albums FROM photos").fetchall()
        album_set: set[str] = set()
        for row in rows:
            raw = row[0]
            if raw:
                try:
                    for a in json.loads(raw):
                        album_set.add(a)
                except (json.JSONDecodeError, TypeError):
                    pass
        return sorted(album_set)

    def get_stats(self) -> dict[str, int]:
        """Return counts of photos and analyses."""
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        photo_count = self._conn.execute(
            "SELECT COUNT(*) FROM photos"
        ).fetchone()[0]
        analysis_count = self._conn.execute(
            "SELECT COUNT(*) FROM photo_analyses"
        ).fetchone()[0]
        return {"photos": photo_count, "analyses": analysis_count}

    # ------------------------------------------------------------------
    # internal
    # ------------------------------------------------------------------

    def _get_cached(self, uuid: str) -> dict[str, Any] | None:
        """Return cached metadata for *uuid*, or ``None``."""
        if not self._conn:
            return None
        row = self._conn.execute(
            "SELECT * FROM photos WHERE uuid = ?", (uuid,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def _insert(self, metadata: dict[str, Any]) -> None:
        """Insert a new photo record."""
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        self._conn.execute(
            """
            INSERT INTO photos
                (uuid, original_filename, date, width, height,
                 latitude, longitude, location_name,
                 albums, tags, description, media_type, file_size,
                 local_id, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            self._metadata_to_row(metadata),
        )

    def _update(self, metadata: dict[str, Any]) -> None:
        """Update an existing photo record."""
        if not self._conn:
            raise RuntimeError("Database not initialised; call init_db() first")

        self._conn.execute(
            """
            UPDATE photos SET
                original_filename = ?,
                date = ?,
                width = ?,
                height = ?,
                latitude = ?,
                longitude = ?,
                location_name = ?,
                albums = ?,
                tags = ?,
                description = ?,
                media_type = ?,
                file_size = ?,
                local_id = ?,
                updated_at = ?
            WHERE uuid = ?
            """,
            (*self._metadata_to_row(metadata)[1:], metadata["uuid"]),
        )

    # ------------------------------------------------------------------
    # serialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_to_row(metadata: dict[str, Any]) -> tuple:
        """Convert a metadata dict into a tuple for SQL insertion."""
        date_val = metadata.get("date")
        date_str = date_val.isoformat() if isinstance(date_val, datetime) else date_val

        return (
            metadata["uuid"],
            metadata.get("original_filename"),
            date_str,
            metadata.get("width"),
            metadata.get("height"),
            metadata.get("latitude"),
            metadata.get("longitude"),
            metadata.get("location_name"),
            json.dumps(metadata.get("albums", [])),
            json.dumps(metadata.get("tags", [])),
            metadata.get("description"),
            metadata.get("media_type"),
            metadata.get("file_size"),
            metadata.get("local_id"),
            datetime.now(tz=None).isoformat(),
        )

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a sqlite3.Row into a plain dict, deserialising JSON fields."""
        d = dict(row)
        for key in ("albums", "tags"):
            raw = d.get(key)
            if isinstance(raw, str):
                try:
                    d[key] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d[key] = []
        return d

    @staticmethod
    def _extract_metadata(photo) -> dict[str, Any]:
        """Pull relevant fields from an osxphotos PhotoInfo object."""
        return {
            "uuid": photo.uuid,
            "original_filename": photo.original_filename,
            "date": photo.date,
            "width": photo.width,
            "height": photo.height,
            "latitude": photo.latitude,
            "longitude": photo.longitude,
            "location_name": photo.place.name if photo.place else None,
            "albums": [a.title for a in photo.albums] if photo.albums else [],
            "tags": list(photo.keywords) if photo.keywords else [],
            "description": photo.description or None,
            "media_type": photo.type.name if photo.type else "photo",
            "file_size": photo.original_filesize if hasattr(photo, "original_filesize") else None,
            "local_id": photo.local_identifier if hasattr(photo, "local_identifier") else None,
        }

    def _count_photos(self) -> int:
        """Return total photo count."""
        if not self._conn:
            return 0
        return self._conn.execute("SELECT COUNT(*) FROM photos").fetchone()[0]
