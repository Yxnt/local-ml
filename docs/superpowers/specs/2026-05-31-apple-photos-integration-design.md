# Apple Photos Integration Design

## Goal

Integrate Apple Photos into the local AI agent, enabling photo search, browsing, metadata queries, and VLM-based image understanding. Deployed on Mac Mini with Photos app.

## Architecture

```
integrations/photos/
├── __init__.py
├── integration.py      # 主集成类 (继承 Integration)
├── indexer.py          # 增量索引器
├── metadata.py         # 元数据查询 (osxphotos)
├── vlm_analyzer.py     # VLM 图片分析 + 缓存
└── tools.py            # Agent 工具定义
```

## Dependencies

- `osxphotos` — Apple Photos 数据访问
- `mlx_vlm` — VLM 图片分析（已有）
- `sqlite-vec` — 向量搜索（已有）

## Integration Interface

```python
class PhotosIntegration(Integration):
    async def connect(self) -> None
    async def disconnect(self) -> None
    async def sync(self) -> dict[str, int]  # 增量索引
    async def query(self, query: str, limit: int = 10) -> list[dict]
    def get_tools(self) -> list[dict]
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
    }
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
      "photo_id": {"type": "string", "description": "Photo ID"},
      "question": {"type": "string", "description": "Question about the photo"}
    },
    "required": ["photo_id"]
  }
}
```

## Incremental Indexing

```python
class PhotosIndexer:
    def __init__(self, db_path: str):
        self._db_path = db_path  # SQLite for metadata cache

    async def sync(self) -> dict[str, int]:
        """Incremental sync using osxphotos."""
        # 1. Query osxphotos for all photos
        # 2. Compare with local cache (by photo UUID + modification date)
        # 3. Insert new / update changed
        # 4. Return stats: {"new": 10, "updated": 5, "total": 1000}
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
    photo_id TEXT PRIMARY KEY,
    analysis TEXT NOT NULL,
    question TEXT,
    created_at TEXT,
    FOREIGN KEY (photo_id) REFERENCES photos(uuid)
);
```

## VLM Analyzer

```python
class VLMAnalyzer:
    def __init__(self, backend, db_path: str):
        self._backend = backend  # ModelBackend for VLM
        self._db_path = db_path  # SQLite for cache

    async def analyze(self, photo_id: str, question: str = None) -> str:
        """Analyze photo with VLM, cache result."""
        # 1. Check cache
        cached = self._get_cached(photo_id, question)
        if cached:
            return cached

        # 2. Get photo path from index
        photo_path = self._get_photo_path(photo_id)

        # 3. Call VLM with image
        prompt = question or "Describe this photo in detail."
        result = await self._backend.generate_with_image(prompt, photo_path)

        # 4. Cache result
        self._cache_result(photo_id, question, result)

        return result
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
    sync_interval: 3600  # seconds
    vlm_model: minicpm-v-4.6  # Model for image analysis
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
PhotosIndexer.query()
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
VLMAnalyzer.analyze()
    ↓
Check cache → miss
    ↓
Load photo from disk
    ↓
VLM generate_with_image(prompt, image_path)
    ↓
Cache result in SQLite
    ↓
Return analysis
```

## Testing Strategy

1. **Unit tests**: Test each component independently
2. **Integration tests**: Test with real Photos library (mock osxphotos)
3. **VLM tests**: Test image analysis with sample images
4. **Cache tests**: Test cache hit/miss behavior

## Known Limitations

1. **macOS only**: osxphotos only works on macOS
2. **Photos app must be installed**: Requires Photos.app
3. **VLM analysis slow**: First analysis takes 2-5 seconds per photo
4. **Large libraries**: Initial sync may take time for large photo collections

## Future Improvements

1. **Face recognition**: Use Photos.app face tags for person search
2. **Semantic search**: Use embeddings for photo descriptions
3. **Batch analysis**: Process multiple photos in parallel
4. **Smart albums**: Auto-create albums based on VLM analysis
