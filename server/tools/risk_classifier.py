"""Conservative risk classification for in-situ tool generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from server.tools.spec import RiskLevel, ToolRequest


@dataclass(frozen=True)
class CapabilityRiskDecision:
    """Decision returned by the explicit missing-capability risk gate."""

    auto_generatable: bool
    risk_level: RiskLevel | None
    reason: str
    blocked_terms: tuple[str, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "auto_generatable": self.auto_generatable,
            "risk_level": self.risk_level.value if self.risk_level else None,
            "reason": self.reason,
            "blocked_terms": list(self.blocked_terms),
        }


class CapabilityRiskClassifier:
    """Allow only clearly local L0/L1 generated tools.

    Model-originated ``ToolRequest`` instances currently default to L0, so this
    classifier derives its decision from the requested capability instead of
    trusting ``ToolRequest.risk_level``.
    """

    _BLOCKED_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "blocked_shell_or_process",
            (
                "bash",
                "command",
                "exec",
                "process",
                "python script",
                "shell",
                "subprocess",
                "terminal",
                "zsh",
            ),
        ),
        (
            "blocked_network",
            (
                "api call",
                "download",
                "fetch url",
                "http",
                "https",
                "internet",
                "network",
                "upload",
                "url",
                "web",
                "webhook",
            ),
        ),
        (
            "blocked_file_mutation",
            (
                "delete file",
                "modify file",
                "move file",
                "project directory",
                "rename file",
                "save file",
                "write file",
            ),
        ),
        (
            "blocked_desktop_control",
            (
                "click",
                "desktop",
                "keyboard",
                "mouse",
                "screen",
                "screenshot",
                "window",
            ),
        ),
        (
            "blocked_credentials",
            (
                "api key",
                "auth",
                "credential",
                "password",
                "secret",
                "token",
            ),
        ),
        (
            "blocked_integration_or_private_data",
            (
                "calendar",
                "contact",
                "email",
                "gmail",
                "imap",
                "mailbox",
                "obsidian",
                "private data",
                "smtp",
            ),
        ),
    )
    _SAFE_HINTS: tuple[str, ...] = (
        "base64",
        "calculate",
        "convert",
        "count",
        "date",
        "dedupe",
        "extract",
        "format",
        "hash",
        "json",
        "markdown",
        "math",
        "normalize",
        "parse",
        "slug",
        "sort",
        "string",
        "text",
    )

    def classify(self, request: ToolRequest) -> CapabilityRiskDecision:
        capability_text = self._capability_text(request)
        if not capability_text.strip() or not request.candidate_name.strip():
            return CapabilityRiskDecision(False, None, "ambiguous_capability")

        full_text = self._request_text(request)
        for reason, terms in self._BLOCKED_GROUPS:
            hits = tuple(term for term in terms if term in full_text)
            if hits:
                return CapabilityRiskDecision(False, None, reason, hits)

        if any(hint in capability_text for hint in self._SAFE_HINTS):
            return CapabilityRiskDecision(True, RiskLevel.L0, "pure_local_computation")

        return CapabilityRiskDecision(False, None, "ambiguous_capability")

    def annotate_request(self, request: ToolRequest) -> CapabilityRiskDecision:
        decision = self.classify(request)
        request.metadata = dict(request.metadata)
        request.metadata["risk_classifier"] = decision.to_metadata()
        if decision.risk_level is not None:
            request.risk_level = decision.risk_level
        return decision

    @staticmethod
    def _capability_text(request: ToolRequest) -> str:
        return " ".join(
            [
                request.candidate_name,
                request.candidate_description,
                request.missing_capability,
            ]
        ).lower()

    @staticmethod
    def _request_text(request: ToolRequest) -> str:
        schema_text = json.dumps(request.candidate_input_schema, ensure_ascii=False, sort_keys=True)
        return " ".join(
            [
                request.candidate_name,
                request.candidate_description,
                request.missing_capability,
                schema_text,
            ]
        ).lower()
