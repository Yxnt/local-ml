"""Tests for Apple Photos indexer."""

import sqlite3
from datetime import datetime

import pytest

from integrations.photos.indexer import PhotosIndexer


@pytest.fixture
def indexer(tmp_path):
    """Create a PhotosIndexer with a temporary database."""
    db_path = str(tmp_path / "photos.db")
    idx = PhotosIndexer(library_path="/tmp/fake.photoslibrary", db_path=db_path)
    idx.init_db()
    yield idx
    idx.close()


def _make_photo(uuid="abc-123", **overrides):
    """Helper to build a photo metadata dict with sensible defaults."""
    base = {
        "uuid": uuid,
        "original_filename": "IMG_0001.HEIC",
        "date": datetime(2024, 6, 15, 10, 30, 0),
        "width": 4032,
        "height": 3024,
        "latitude": 37.7749,
        "longitude": -122.4194,
        "location_name": "San Francisco",
        "albums": ["Vacation", "Favorites"],
        "tags": ["sunset", "beach", "golden-gate"],
        "description": "Golden Gate Bridge at sunset",
        "media_type": "photo",
        "file_size": 3_500_000,
    }
    base.update(overrides)
    return base


class TestPhotosIndexer:
    """Test suite for PhotosIndexer."""

    def test_init_creates_database(self, tmp_path):
        """init_db() should create the database file and required tables."""
        db_path = str(tmp_path / "photos.db")
        idx = PhotosIndexer(library_path="/tmp/fake.photoslibrary", db_path=db_path)
        idx.init_db()

        # Verify database file exists
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()

        assert "photos" in tables
        assert "photo_analyses" in tables

        idx.close()

    def test_insert_and_get(self, indexer):
        """_insert() should persist metadata; _get_cached() should retrieve it."""
        photo = _make_photo()
        indexer._insert(photo)

        cached = indexer._get_cached("abc-123")
        assert cached is not None
        assert cached["uuid"] == "abc-123"
        assert cached["original_filename"] == "IMG_0001.HEIC"
        assert cached["latitude"] == pytest.approx(37.7749)
        assert cached["longitude"] == pytest.approx(-122.4194)

    def test_search_by_location(self, indexer):
        """search_sync() should find photos by location_name."""
        indexer._insert(_make_photo(uuid="loc-1", location_name="San Francisco"))
        indexer._insert(_make_photo(uuid="loc-2", location_name="Tokyo"))
        indexer._insert(_make_photo(uuid="loc-3", location_name="San Francisco"))

        results = indexer.search_sync(query="San Francisco")
        assert len(results) == 2
        uuids = {r["uuid"] for r in results}
        assert uuids == {"loc-1", "loc-3"}

    def test_search_by_date_range(self, indexer):
        """search_sync() should filter by date_from / date_to."""
        indexer._insert(
            _make_photo(uuid="d1", date=datetime(2024, 1, 10))
        )
        indexer._insert(
            _make_photo(uuid="d2", date=datetime(2024, 6, 15))
        )
        indexer._insert(
            _make_photo(uuid="d3", date=datetime(2024, 12, 25))
        )

        results = indexer.search_sync(
            date_from=datetime(2024, 3, 1),
            date_to=datetime(2024, 9, 1),
        )
        assert len(results) == 1
        assert results[0]["uuid"] == "d2"

    def test_search_by_tags(self, indexer):
        """search_sync() should match photos whose tags contain the query."""
        indexer._insert(
            _make_photo(uuid="t1", tags=["sunset", "beach"], description="")
        )
        indexer._insert(
            _make_photo(uuid="t2", tags=["mountain", "hiking"], description="")
        )
        indexer._insert(
            _make_photo(uuid="t3", tags=["sunset", "cityscape"], description="")
        )

        results = indexer.search_sync(query="sunset")
        assert len(results) == 2
        uuids = {r["uuid"] for r in results}
        assert uuids == {"t1", "t3"}
