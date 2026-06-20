"""Generate and save local-first daily journals."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from evolution.journal_style_adapter import style_from_preferences
from evolution.prompt_optimizer import build_journal_prompt
from lifelog.event_builder import Event
from memory.memory_schema import EventMemory, PreferenceMemory

LLMClient = Callable[[str], str]


@dataclass(frozen=True)
class JournalDraft:
    markdown: str
    prompt: str


def generate_journal(
    events: list[Event],
    *,
    journal_date: date,
    llm_client: LLMClient | None = None,
    preferences: list[PreferenceMemory] | None = None,
    similar_events: list[EventMemory] | None = None,
    warnings: list[str] | None = None,
) -> str:
    return prepare_journal(
        events,
        journal_date=journal_date,
        llm_client=llm_client,
        preferences=preferences,
        similar_events=similar_events,
        warnings=warnings,
    ).markdown


def prepare_journal(
    events: list[Event],
    *,
    journal_date: date,
    llm_client: LLMClient | None = None,
    preferences: list[PreferenceMemory] | None = None,
    similar_events: list[EventMemory] | None = None,
    warnings: list[str] | None = None,
) -> JournalDraft:
    preference_list = preferences or []
    similar_list = similar_events or []
    prompt = build_journal_prompt(
        events=events,
        journal_date=journal_date,
        preferences=preference_list,
        similar_events=similar_list,
    )
    client = llm_client if llm_client is not None else _default_local_llm_client()
    if client is not None and events:
        try:
            text = client(prompt).strip()
            if text.startswith("# Daily Journal"):
                return JournalDraft(markdown=_append_warnings(text, warnings), prompt=prompt)
        except Exception:
            pass
    return JournalDraft(
        markdown=_append_warnings(
            _template_journal(events, journal_date, preferences=preference_list),
            warnings,
        ),
        prompt=prompt,
    )


def save_journal(markdown: str, output_dir: str | Path, journal_date: date) -> Path:
    root = Path(output_dir).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{journal_date.isoformat()}.md"
    path.write_text(markdown, encoding="utf-8")
    return path


def _default_local_llm_client() -> LLMClient | None:
    if os.environ.get("LIFELOG_DISABLE_LLM") == "1":
        return None
    openai_client = _openai_compatible_client()
    if openai_client is not None:
        return openai_client
    return _ollama_client()


def _ollama_client() -> LLMClient | None:
    ollama_url = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    model = os.environ.get("LIFELOG_OLLAMA_MODEL", "llama3.2")

    def call_ollama(prompt: str) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0.3}}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{ollama_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data.get("response", ""))

    try:
        urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=1).close()
    except (OSError, urllib.error.URLError):
        return None
    return call_ollama


def _openai_compatible_client() -> LLMClient | None:
    base_url = os.environ.get("LIFELOG_OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
    model = os.environ.get("LIFELOG_OPENAI_MODEL", os.environ.get("DEFAULT_MODEL", "gemma-4-e2b-it-4bit"))

    def call_openai_compatible(prompt: str) -> str:
        payload = json.dumps(
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "temperature": 0.3,
                "max_tokens": 900,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"].get("content") or "")

    try:
        urllib.request.urlopen(f"{base_url}/models", timeout=1).close()
    except (OSError, urllib.error.URLError):
        return None
    return call_openai_compatible


def _template_journal(
    events: list[Event],
    journal_date: date,
    *,
    preferences: list[PreferenceMemory] | None = None,
) -> str:
    style = style_from_preferences(preferences or [])
    buckets = {
        "Morning": [event for event in events if event.start_time.hour < 12],
        "Afternoon": [event for event in events if 12 <= event.start_time.hour < 18],
        "Evening": [event for event in events if event.start_time.hour >= 18],
    }
    lines = [f"# Daily Journal - {journal_date.isoformat()}", ""]
    for section, section_events in buckets.items():
        lines.extend([f"## {section}", ""])
        if not section_events:
            lines.extend(["No photos were found for this part of the day.", ""])
            continue
        for event in section_events:
            signals = ", ".join(event.summary_signals) if event.summary_signals else "quiet moments"
            count = len(event.photos)
            noun = "photo" if count == 1 else "photos"
            prefix = _emphasis_prefix(event, style)
            lines.append(
                f"- {prefix}{event.start_time.strftime('%-I:%M %p')}-{event.end_time.strftime('%-I:%M %p')}: "
                f"{count} {noun} suggested {signals}."
            )
            if not style.concise:
                lines.append(_event_sentence(event))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _event_sentence(event: Event) -> str:
    if not event.summary_signals:
        return "  The images mark a small moment in the day without much extra context."
    main = event.summary_signals[0]
    return f"  It reads like a {main} moment, with the photos giving the day a little shape."


def _emphasis_prefix(event: Event, style) -> str:  # type: ignore[no-untyped-def]
    signals = " ".join(event.summary_signals).lower()
    if style.emphasize_health and any(word in signals for word in ["gym", "fitness", "workout", "health"]):
        return "Health: "
    if style.emphasize_work and any(word in signals for word in ["work", "office", "laptop", "productivity"]):
        return "Work: "
    if style.emphasize_social and any(word in signals for word in ["people", "friend", "family", "social"]):
        return "Social: "
    return ""


def _append_warnings(markdown: str, warnings: list[str] | None) -> str:
    clean_warnings = [warning for warning in warnings or [] if warning.strip()]
    if not clean_warnings:
        return markdown if markdown.endswith("\n") else markdown + "\n"
    lines = [markdown.rstrip(), "", "## Pipeline Notes", ""]
    lines.extend(f"- {warning}" for warning in clean_warnings)
    return "\n".join(lines).rstrip() + "\n"
