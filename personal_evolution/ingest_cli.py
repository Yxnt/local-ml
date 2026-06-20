from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from datetime import date
from pathlib import Path
from typing import Sequence

from lifelog.photos_reader import PhotoItem
from personal_evolution.pipeline import (
    PersonalIngestionResult,
    ingest_personal_sources,
)
from personal_evolution.store import PersonalEvolutionStore
from server.config import load_config


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    target_date = date.fromisoformat(args.date) if args.date else date.today()
    config = load_config(args.config, reload=True)
    if args.diagnose_sources:
        _diagnose_sources(config)
        return 0

    db_path = Path(
        args.db
        or os.environ.get("PERSONAL_EVOLUTION_DB", "memory/personal_evolution.sqlite3")
    ).expanduser()
    vaults = _obsidian_vaults(args.obsidian_vault, config.integrations.obsidian.vaults)
    read_photos = _no_photos_reader if args.no_photos else None

    result = ingest_personal_sources(
        PersonalEvolutionStore(db_path),
        obsidian_vaults=vaults,
        target_date=target_date,
        read_photos=read_photos,
    )

    print(
        "personal ingestion complete: "
        f"evidence={result.evidence_saved} "
        f"events={result.events_saved} "
        f"candidates={result.candidates_saved}"
    )
    for error in result.errors:
        print(f"warning: {error}")
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest local personal sources into the review-first memory store."
    )
    parser.add_argument("--config", help="Path to config.yaml. Defaults to project config.")
    parser.add_argument("--db", help="Personal evolution SQLite DB path.")
    parser.add_argument("--date", help="Target date for Photos ingestion, YYYY-MM-DD.")
    parser.add_argument(
        "--obsidian-vault",
        action="append",
        default=[],
        help="Obsidian vault path. May be passed multiple times; overrides config vaults.",
    )
    parser.add_argument(
        "--no-photos",
        action="store_true",
        help="Skip Photos ingestion. Useful for smoke tests on machines without Photos access.",
    )
    parser.add_argument(
        "--diagnose-sources",
        action="store_true",
        help="Print source availability checks and exit without ingesting.",
    )
    return parser


def _obsidian_vaults(
    explicit_vaults: list[str],
    configured_vaults: dict[str, str],
) -> list[Path]:
    source = explicit_vaults or list(configured_vaults.values())
    return [Path(vault).expanduser() for vault in source]


def _no_photos_reader(_target_date: date) -> list[PhotoItem]:
    return []


def _diagnose_sources(config) -> None:  # type: ignore[no-untyped-def]
    for name, vault in config.integrations.obsidian.vaults.items():
        path = Path(vault).expanduser()
        print(f"obsidian:{name} exists={path.exists()} path={path}")

    photos_library = Path(config.integrations.photos.photos_library).expanduser()
    print(f"photos_library exists={photos_library.exists()} path={photos_library}")
    for module_name in ("osxphotos", "Photos", "objc"):
        print(f"module:{module_name} available={_module_available(module_name)}")
    print(f"osascript available={shutil.which('osascript') is not None}")


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


if __name__ == "__main__":
    raise SystemExit(main())
