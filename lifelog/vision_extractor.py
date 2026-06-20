"""Extract local vision signals with Apple Vision or harmless heuristics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from lifelog.photos_reader import PhotoItem


@dataclass(frozen=True)
class VisionResult:
    photo_path: str
    objects: list[str]
    scene: str | None
    people: list[str]


def extract_vision(photo: PhotoItem) -> VisionResult:
    try:
        result = _extract_with_apple_vision(photo)
        if result is not None:
            return result
    except Exception:
        pass
    return _extract_with_heuristics(photo)


def extract_all_vision(photos: list[PhotoItem]) -> dict[str, VisionResult]:
    return {photo.path: extract_vision(photo) for photo in photos}


def _extract_with_apple_vision(photo: PhotoItem) -> VisionResult | None:
    try:
        import Foundation  # type: ignore
        import Vision  # type: ignore
    except Exception:
        return None

    url = Foundation.NSURL.fileURLWithPath_(photo.path)
    handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, {})

    labels: list[str] = []
    face_count = 0

    def labels_done(request, error) -> None:  # type: ignore[no-untyped-def]
        if error is not None:
            return
        for observation in request.results() or []:
            try:
                identifier = str(observation.identifier()).replace("_", " ").lower()
                confidence = float(observation.confidence())
                if confidence >= 0.35 and identifier:
                    labels.append(identifier)
            except Exception:
                continue

    def faces_done(request, error) -> None:  # type: ignore[no-untyped-def]
        nonlocal face_count
        if error is None:
            face_count = len(request.results() or [])

    requests = []
    if hasattr(Vision, "VNClassifyImageRequest"):
        requests.append(Vision.VNClassifyImageRequest.alloc().initWithCompletionHandler_(labels_done))
    if hasattr(Vision, "VNDetectFaceRectanglesRequest"):
        requests.append(Vision.VNDetectFaceRectanglesRequest.alloc().initWithCompletionHandler_(faces_done))
    if not requests:
        return None

    ok = handler.performRequests_error_(requests, None)
    if not ok:
        return None

    people = ["face"] if face_count else []
    scene = labels[0] if labels else None
    objects = _dedupe(labels[1:] if scene else labels)
    return VisionResult(photo_path=photo.path, objects=objects, scene=scene, people=people)


def _extract_with_heuristics(photo: PhotoItem) -> VisionResult:
    stem = Path(photo.path).stem.replace("_", " ").replace("-", " ").lower()
    words = [word for word in stem.split() if len(word) > 2 and not word.isdigit()]
    people = [word for word in words if word in {"person", "people", "portrait", "selfie", "face"}]
    scene = _guess_scene(words)
    objects = [word for word in words if word not in set(people)]
    if scene and scene not in objects:
        objects.append(scene)
    return VisionResult(
        photo_path=photo.path,
        objects=_dedupe(objects[:8]),
        scene=scene,
        people=_dedupe(people),
    )


def _guess_scene(words: list[str]) -> str | None:
    scene_keywords = {
        "coffee": "cafe",
        "cafe": "cafe",
        "lunch": "restaurant",
        "dinner": "restaurant",
        "food": "restaurant",
        "work": "working session",
        "office": "working session",
        "commute": "commute",
        "train": "commute",
        "walk": "walk",
        "home": "home",
    }
    for word in words:
        if word in scene_keywords:
            return scene_keywords[word]
    return None


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = value.strip().lower()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result
