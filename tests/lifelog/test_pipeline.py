from __future__ import annotations

from datetime import date

from lifelog.pipeline import run_pipeline


def test_run_pipeline_writes_markdown_even_when_reader_fails(tmp_path) -> None:
    def broken_reader(target_date: date):
        raise RuntimeError("Photos access denied")

    output_path = run_pipeline(
        target_date=date(2026, 6, 20),
        output_dir=tmp_path,
        read_photos=broken_reader,
    )

    assert output_path == tmp_path / "2026-06-20.md"
    content = output_path.read_text(encoding="utf-8")
    assert content.startswith("# Daily Journal - 2026-06-20")
    assert "Photos access denied" in content
