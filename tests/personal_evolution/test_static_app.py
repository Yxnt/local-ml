from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = ROOT / "web" / "personal-evolution"


def test_static_review_console_contains_required_sections_and_api_hooks():
    html = (APP_DIR / "index.html").read_text(encoding="utf-8")
    css = (APP_DIR / "styles.css").read_text(encoding="utf-8")
    js = (APP_DIR / "app.js").read_text(encoding="utf-8")

    assert "Review Today/Week" in html
    assert "Learning Queue" in html
    assert "Memory Ledger" in html

    assert "@media" in css

    assert "/personal-evolution/candidates" in js
    assert "/personal-evolution/memories" in js
    assert "/personal-evolution/review/" in js
    assert "approveCandidate" in js
    assert "rejectCandidate" in js
    assert "revokeMemory" in js
    assert "candidate_id" in js
    assert "memory_id" in js
    assert "claim" in js
    assert "memory_type" in js
    assert "rationale" in js
    assert "content" in js
    assert "approved_at" in js
    assert "status" in js
    assert "events" in js
    assert "evidence" in js
    assert "review.date" in js
    assert "body: JSON.stringify({})" in js

    lower_js = js.lower()
    dangerous_patterns = [
        ".innerhtml",
        ".outerhtml",
        "insertadjacenthtml",
        "document.write",
        "onclick",
        "onerror",
    ]
    for pattern in dangerous_patterns:
        assert pattern not in lower_js

    assert "createElement" in js
    assert "textContent" in js
    assert "replaceChildren" in js
    assert "addEventListener" in js
