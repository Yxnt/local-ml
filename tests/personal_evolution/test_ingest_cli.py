from __future__ import annotations

from datetime import date
from pathlib import Path

import personal_evolution.ingest_cli as ingest_cli


def test_ingest_cli_loads_config_and_passes_vaults_and_db(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    config = tmp_path / "config.yaml"
    db_path = tmp_path / "personal.sqlite3"
    config.write_text(
        f"""
integrations:
  obsidian:
    vaults:
      main: {vault}
""",
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_ingest(store, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(
            {
                "db_path": store.db_path,
                "obsidian_vaults": kwargs["obsidian_vaults"],
                "target_date": kwargs["target_date"],
                "read_photos": kwargs["read_photos"],
            }
        )
        return ingest_cli.PersonalIngestionResult(
            evidence_saved=1,
            events_saved=1,
            candidates_saved=1,
            errors=[],
        )

    monkeypatch.setattr(ingest_cli, "ingest_personal_sources", fake_ingest)

    exit_code = ingest_cli.main(
        [
            "--config",
            str(config),
            "--db",
            str(db_path),
            "--date",
            "2026-06-20",
            "--no-photos",
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "db_path": db_path,
            "obsidian_vaults": [vault],
            "target_date": date(2026, 6, 20),
            "read_photos": ingest_cli._no_photos_reader,
        }
    ]
    assert "evidence=1 events=1 candidates=1" in capsys.readouterr().out


def test_ingest_cli_allows_explicit_obsidian_vault_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    explicit_vault = tmp_path / "explicit"
    explicit_vault.mkdir()
    config = tmp_path / "empty-config.yaml"
    config.write_text("integrations: {}\n", encoding="utf-8")
    calls: list[list[Path]] = []

    def fake_ingest(store, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(kwargs["obsidian_vaults"])
        return ingest_cli.PersonalIngestionResult(0, 0, 0, [])

    monkeypatch.setattr(ingest_cli, "ingest_personal_sources", fake_ingest)

    exit_code = ingest_cli.main(
        [
            "--config",
            str(config),
            "--obsidian-vault",
            str(explicit_vault),
            "--no-photos",
        ]
    )

    assert exit_code == 0
    assert calls == [[explicit_vault]]


def test_ingest_cli_can_diagnose_configured_sources(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    vault = tmp_path / "vault"
    vault.mkdir()
    photos_library = tmp_path / "Photos Library.photoslibrary"
    photos_library.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(
        f"""
integrations:
  obsidian:
    vaults:
      main: {vault}
  photos:
    photos_library: {photos_library}
""",
        encoding="utf-8",
    )

    monkeypatch.setattr(ingest_cli, "_module_available", lambda name: name == "osxphotos")
    monkeypatch.setattr(ingest_cli.shutil, "which", lambda name: "/usr/bin/osascript")

    exit_code = ingest_cli.main(["--config", str(config), "--diagnose-sources"])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert f"obsidian:main exists=True path={vault}" in output
    assert f"photos_library exists=True path={photos_library}" in output
    assert "module:osxphotos available=True" in output
    assert "module:Photos available=False" in output
    assert "osascript available=True" in output
