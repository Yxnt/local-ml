"""Read today's photos from macOS Photos with safe local fallbacks."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PhotoItem:
    path: str
    timestamp: datetime
    local_identifier: str


def read_today_photos(
    target_date: date | None = None,
    *,
    export_dir: str | Path | None = None,
) -> list[PhotoItem]:
    """Return photos from *target_date* using Photos.framework or AppleScript.

    The function never raises for normal platform/permission failures. It returns
    an empty list when Photos access is unavailable so the pipeline can still
    create a journal.
    """

    target = target_date or datetime.now().date()
    for reader in (_read_with_photos_framework, _read_with_applescript_export):
        try:
            items = reader(target, export_dir=export_dir)
        except Exception:
            items = []
        if items:
            return sorted(items, key=lambda item: item.timestamp)
    return []


def _read_with_photos_framework(
    target_date: date,
    *,
    export_dir: str | Path | None = None,
) -> list[PhotoItem]:
    try:
        import Photos  # type: ignore
        import objc  # type: ignore  # noqa: F401
    except Exception:
        return []

    start = datetime.combine(target_date, time.min)
    end = start + timedelta(days=1)

    fetch_options = Photos.PHFetchOptions.alloc().init()
    predicate = _ns_predicate_for_date_range(start, end)
    if predicate is not None:
        fetch_options.setPredicate_(predicate)

    assets = Photos.PHAsset.fetchAssetsWithMediaType_options_(
        Photos.PHAssetMediaTypeImage,
        fetch_options,
    )
    items: list[PhotoItem] = []
    export_root = _export_root(export_dir)

    for index in range(int(assets.count())):
        asset = assets.objectAtIndex_(index)
        timestamp = _to_datetime(asset.creationDate()) or start
        if not start <= timestamp < end:
            continue
        identifier = str(asset.localIdentifier())
        path = _export_asset(asset, export_root, identifier)
        if path:
            items.append(PhotoItem(path=path, timestamp=timestamp, local_identifier=identifier))

    return items


def _ns_predicate_for_date_range(start: datetime, end: datetime) -> Any | None:
    try:
        import Foundation  # type: ignore

        start_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(start.timestamp())
        end_date = Foundation.NSDate.dateWithTimeIntervalSince1970_(end.timestamp())
        return Foundation.NSPredicate.predicateWithFormat_argumentArray_(
            "creationDate >= %@ AND creationDate < %@",
            [start_date, end_date],
        )
    except Exception:
        return None


def _export_asset(asset: Any, export_root: Path, identifier: str) -> str | None:
    """Export a PHAsset to disk best-effort.

    PyObjC Photos export APIs vary by macOS/PyObjC version. Keep this contained:
    if anything is unavailable, return None and let AppleScript/fallbacks handle
    the rest.
    """

    try:
        import Photos  # type: ignore
        import Foundation  # type: ignore

        safe_id = "".join(ch if ch.isalnum() else "_" for ch in identifier)[:80]
        destination = export_root / f"{safe_id}.jpg"
        if destination.exists():
            return str(destination)

        resources = Photos.PHAssetResource.assetResourcesForAsset_(asset)
        if int(resources.count()) == 0:
            return None
        resource = resources.objectAtIndex_(0)
        options = Photos.PHAssetResourceRequestOptions.alloc().init()
        done: dict[str, Any] = {"finished": False, "error": None}

        def completion(error: Any) -> None:
            done["finished"] = True
            done["error"] = error

        manager = Photos.PHAssetResourceManager.defaultManager()
        url = Foundation.NSURL.fileURLWithPath_(str(destination))
        manager.writeDataForAssetResource_toFile_options_completionHandler_(
            resource,
            url,
            options,
            completion,
        )

        # Give Photos a short synchronous window. If it does not finish, the
        # caller can still continue with AppleScript or no-photo fallback.
        import time as time_module

        deadline = time_module.time() + 5
        while not done["finished"] and time_module.time() < deadline:
            time_module.sleep(0.05)

        if destination.exists() and done["error"] is None:
            return str(destination)
    except Exception:
        return None
    return None


def _read_with_applescript_export(
    target_date: date,
    *,
    export_dir: str | Path | None = None,
) -> list[PhotoItem]:
    if shutil.which("osascript") is None:
        return []

    export_root = _export_root(export_dir)
    start_text = target_date.strftime("%Y-%m-%d 00:00:00")
    end_text = (target_date + timedelta(days=1)).strftime("%Y-%m-%d 00:00:00")
    script = f"""
set exportFolder to POSIX file "{str(export_root)}"
set startDate to date "{start_text}"
set endDate to date "{end_text}"
set outputLines to {{}}
tell application "Photos"
    set todaysItems to every media item whose date is greater than or equal to startDate and date is less than endDate
    repeat with itemRef in todaysItems
        try
            set itemId to id of itemRef
            set itemDate to date of itemRef
            export {{itemRef}} to exportFolder with using originals
            set end of outputLines to itemId & tab & (itemDate as string)
        end try
    end repeat
end tell
set AppleScript's text item delimiters to linefeed
return outputLines as text
"""
    try:
        before = {p.resolve() for p in export_root.glob("*") if p.is_file()}
        result = subprocess.run(
            ["osascript", "-e", script],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    exported = sorted(
        (p for p in export_root.glob("*") if p.is_file() and p.resolve() not in before),
        key=lambda path: path.stat().st_mtime,
    )
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    items: list[PhotoItem] = []
    for index, path in enumerate(exported):
        identifier = rows[index].split("\t", 1)[0] if index < len(rows) else path.stem
        timestamp = datetime.fromtimestamp(path.stat().st_mtime)
        items.append(PhotoItem(str(path), timestamp, identifier))
    return items


def _export_root(export_dir: str | Path | None) -> Path:
    if export_dir is None:
        root = Path(tempfile.gettempdir()) / "local-ml-lifelog-photos"
    else:
        root = Path(export_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    return root


def _to_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    try:
        timestamp = value.timeIntervalSince1970()
        return datetime.fromtimestamp(float(timestamp))
    except Exception:
        pass
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value)
    return None


def default_photos_export_dir() -> Path:
    return Path(os.environ.get("LIFELOG_PHOTOS_EXPORT_DIR", "~/local-ml-journal/photos")).expanduser()
