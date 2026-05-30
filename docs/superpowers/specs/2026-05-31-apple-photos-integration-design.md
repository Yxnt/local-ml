# Apple Photos Integration Design

## Goal

Integrate Apple Photos into the local AI agent, enabling photo search, browsing, metadata queries, and VLM-based image understanding. Deployed on Mac Mini with Photos app.

## Architecture

```
integrations/photos/
├── __init__.py
├── integration.py      # 主集成类 (继承 Integration)
├── indexer.py          # 增量索引器 (osxphotos)
├── vlm_analyzer.py     # VLM 图片分析 + 缓存
└── tools.py            # Agent 工具定义
```

## Dependencies

- `osxphotos` — Apple Photos 数据访问
- `mlx_vlm` — VLM 图片分析（已有）
- `sqlite3` — 元数据缓存（已有）

## Integration Interface

```python
class PhotosIntegration(Integration):
    """Apple Photos integration.

    Components:
    - indexer: PhotosIndexer (sync metadata from Photos.app)
    - vlm: VLMAnalyzer (image understanding with caching)
    """

    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self._indexer: PhotosIndexer | None = None
        self._vlm: VLMAnalyzer | None = None

    async def connect(self) -> None:
        """Initialize indexer and VLM analyzer."""
        library_path = self.config.config.get("library_path")
        db_path = self.config.config.get("db_path", "memory/photos.db")
        self._indexer = PhotosIndexer(library_path, db_path)
        # VLM is optional - requires model to be loaded
        vlm_model = self.config.config.get("vlm_model")
        if vlm_model:
            self._vlm = VLMAnalyzer(vlm_model, db_path)

    async def disconnect(self) -> None:
        """Cleanup."""
        self._indexer = None
        self._vlm = None

    async def sync(self) -> dict[str, int]:
        """Incremental sync metadata from Photos.app."""
        return await self._indexer.sync()

    async def query(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search photos by metadata."""
        return await self._indexer.search(query, limit)

    def get_tools(self) -> list[dict[str, Any]]:
        """Return tool definitions for the agent."""
        return [
            self._make_search_tool(),
            self._make_list_albums_tool(),
            self._make_describe_tool(),
        ]
```

## Tools

### photos_search

```json
{
  "name": "photos_search",
  "description": "Search photos by time, location, tags, or description.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string", "description": "Search keyword"},
      "date_from": {"type": "string", "description": "Start date (YYYY-MM-DD)"},
      "date_to": {"type": "string", "description": "End date (YYYY-MM-DD)"},
      "album": {"type": "string", "description": "Album name"},
      "limit": {"type": "integer", "description": "Max results (default: 10)"}
    },
    "required": []
  }
}
```

### photos_list_albums

```json
{
  "name": "photos_list_albums",
  "description": "List all photo albums.",
  "parameters": {"type": "object", "properties": {}}
}
```

### photos_describe

```json
{
  "name": "photos_describe",
  "description": "Analyze photo content using VLM.",
  "parameters": {
    "type": "object",
    "properties": {
      "photo_id": {"type": "string", "description": "Photo UUID from search results"},
      "question": {"type": "string", "description": "Question about the photo (optional)"}
    },
    "required": ["photo_id"]
  }
}
```

## Incremental Indexing

```python
class PhotosIndexer:
    """Incremental indexer using osxphotos."""

    def __init__(self, library_path: str, db_path: str):
        self._library_path = library_path
        self._db_path = db_path  # SQLite for metadata cache

    async def sync(self) -> dict[str, int]:
        """Incremental sync using osxphotos."""
        import osxphotos

        photosdb = osxphotos.PhotosDB(self._library_path)

        new_count = 0
        updated_count = 0

        for photo in photosdb.photos():
            # Check if already indexed
            existing = self._get_cached(photo.uuid)
            if existing and existing["modified_at"] == str(photo.date_modified):
                continue  # Skip unchanged

            # Extract metadata
            metadata = {
                "uuid": photo.uuid,
                "path": photo.path,
                "filename": photo.filename,
                "created_at": str(photo.date),
                "modified_at": str(photo.date_modified),
                "latitude": photo.latitude,
                "longitude": photo.longitude,
                "location_name": photo.place.name if photo.place else None,
                "album": ", ".join(a.title for a in photo.albums),
                "description": photo.description,
                "tags": json.dumps(photo.keywords),
            }

            if existing:
                self._update(metadata)
                updated_count += 1
            else:
                self._insert(metadata)
                new_count += 1

        return {"new": new_count, "updated": updated_count, "total": len(photosdb.photos())}

    async def search(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search photos by metadata."""
        # Search in filename, description, location, tags
        sql = """
            SELECT * FROM photos
            WHERE filename LIKE ? OR description LIKE ?
               OR location_name LIKE ? OR tags LIKE ?
            ORDER BY created_at DESC
            LIMIT ?
        """
        pattern = f"%{query}%"
        # Execute and return results
```

### Metadata Schema

```sql
CREATE TABLE IF NOT EXISTS photos (
    uuid TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    filename TEXT,
    created_at TEXT,
    modified_at TEXT,
    latitude REAL,
    longitude REAL,
    location_name TEXT,
    album TEXT,
    description TEXT,
    tags TEXT,  -- JSON array
    has_vlm_analysis BOOLEAN DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS photo_analyses (
    photo_id TEXT NOT NULL,
    question TEXT DEFAULT '',
    analysis TEXT NOT NULL,
    created_at TEXT,
    PRIMARY KEY (photo_id, question),
    FOREIGN KEY (photo_id) REFERENCES photos(uuid)
);
```

## VLM Analyzer

```python
class VLMAnalyzer:
    """VLM-based image analysis with caching."""

    def __init__(self, model_name: str, db_path: str):
        self._model_name = model_name
        self._db_path = db_path
        self._backend = None  # Lazy-loaded from ModelRegistry

    async def analyze(self, photo_id: str, question: str = None) -> str:
        """Analyze photo with VLM, cache result."""
        # 1. Check cache
        cached = self._get_cached(photo_id, question or "")
        if cached:
            return cached

        # 2. Get photo path from index
        photo_path = self._get_photo_path(photo_id)
        if not photo_path:
            return "Photo not found"

        # 3. Load VLM backend (lazy)
        if not self._backend:
            from backends.registry import ModelRegistry
            registry = ModelRegistry()
            registry.register_defaults()
            self._backend = await registry.get_or_load(self._model_name)

        # 4. Call VLM with image
        prompt = question or "Describe this photo in detail."
        # Note: ModelBackend needs generate_with_image() method
        # For now, use generate() with image path in prompt
        result = self._backend.generate(
            prompt=f"{prompt}\n\n[Image: {photo_path}]",
            max_tokens=512,
        )

        # 5. Cache result
        self._cache_result(photo_id, question or "", result)

        return result

    def _get_cached(self, photo_id: str, question: str) -> str | None:
        """Get cached analysis."""
        # Query photo_analyses table

    def _cache_result(self, photo_id: str, question: str, analysis: str) -> None:
        """Cache analysis result."""
        # Insert into photo_analyses table
```

## Privacy Strategy

| Data | Local | Remote (sanitized) |
|------|-------|-------------------|
| Photo metadata | ✅ | ✅ (after sanitization) |
| Photo path | ✅ | ❌ |
| VLM analysis | ✅ | ❌ |
| Photo itself | ✅ | ❌ |

## Configuration

```yaml
# config.yaml
integrations:
  photos:
    enabled: true
    library_path: ~/Pictures/Photos Library.photoslibrary
    db_path: memory/photos.db
    sync_interval: 3600  # seconds
    vlm_model: minicpm-v-4.6  # Model for image analysis (optional)
    cache_analyses: true
    max_results: 20
```

## Data Flow

### Search Flow

```
User: "找一下去年在东京拍的照片"
    ↓
photos_search(query="东京", date_from="2025-01-01", date_to="2025-12-31")
    ↓
PhotosIntegration.query()
    ↓
PhotosIndexer.search()
    ↓
SQLite: SELECT * FROM photos WHERE location_name LIKE '%东京%' AND created_at BETWEEN ...
    ↓
Return photo list with metadata
```

### VLM Analysis Flow

```
User: "有猫的照片是哪张？"
    ↓
photos_describe(photo_id="xxx", question="这张照片里有猫吗？")
    ↓
PhotosIntegration → VLMAnalyzer.analyze()
    ↓
Check cache → miss
    ↓
Load photo from disk
    ↓
VLM generate(prompt with image path)
    ↓
Cache result in SQLite
    ↓
Return analysis
```

## Testing Strategy

1. **Unit tests**: Test each component independently (mock osxphotos)
2. **Integration tests**: Test with sample photos
3. **VLM tests**: Test image analysis with sample images
4. **Cache tests**: Test cache hit/miss behavior

## Known Limitations

1. **macOS only**: osxphotos only works on macOS
2. **Photos app must be installed**: Requires Photos.app
3. **VLM analysis slow**: First analysis takes 2-5 seconds per photo
4. **Large libraries**: Initial sync may take time for large photo collections
5. **No generate_with_image()**: Current ModelBackend doesn't support image input directly

## Future Improvements

1. **Face recognition**: Use Photos.app face tags for person search
2. **Semantic search**: Use embeddings for photo descriptions
3. **Batch analysis**: Process multiple photos in parallel
4. **Smart albums**: Auto-create albums based on VLM analysis
5. **Image input support**: Add `generate_with_image()` to ModelBackend
