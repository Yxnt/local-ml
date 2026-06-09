from server.tools.risk_classifier import CapabilityRiskClassifier
from server.tools.spec import RiskLevel, ToolRequest


def _request(name: str, capability: str, schema: dict | None = None) -> ToolRequest:
    return ToolRequest(
        candidate_name=name,
        candidate_description=capability,
        missing_capability=capability,
        candidate_input_schema=schema
        or {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    )


def test_allows_pure_text_processing_as_l0():
    decision = CapabilityRiskClassifier().classify(
        _request("text_count_words", "Count words in a provided text string")
    )

    assert decision.auto_generatable is True
    assert decision.risk_level == RiskLevel.L0
    assert decision.reason == "pure_local_computation"


def test_allows_json_date_and_math_transformations():
    classifier = CapabilityRiskClassifier()

    requests = [
        _request("json_normalize", "Normalize JSON keys in a provided JSON object"),
        _request("date_format", "Convert a provided date string to ISO format"),
        _request("math_percent", "Calculate a percentage from numeric inputs"),
    ]

    for req in requests:
        decision = classifier.classify(req)
        assert decision.auto_generatable is True
        assert decision.risk_level == RiskLevel.L0
        assert decision.reason == "pure_local_computation"


def test_blocks_empty_or_ambiguous_missing_capability_even_with_default_l0():
    decision = CapabilityRiskClassifier().classify(_request("do_the_thing", ""))

    assert decision.auto_generatable is False
    assert decision.risk_level is None
    assert decision.reason == "ambiguous_capability"


def test_blocks_network_file_shell_and_integration_requests():
    classifier = CapabilityRiskClassifier()

    blocked = [
        _request("download_url", "Download a URL from the internet"),
        _request("write_file", "Write a file to the project directory"),
        _request("run_shell", "Run a shell command"),
        _request("calendar_read", "Read the user's calendar"),
        _request("desktop_click", "Click a button on the desktop"),
        _request("token_lookup", "Use an API key to call a service"),
    ]

    for req in blocked:
        decision = classifier.classify(req)
        assert decision.auto_generatable is False
        assert decision.risk_level is None
        assert decision.reason.startswith("blocked_")


def test_annotate_request_records_classifier_metadata_without_trusting_default_risk():
    req = _request("calendar_read", "Read the user's calendar")

    decision = CapabilityRiskClassifier().annotate_request(req)

    assert decision.auto_generatable is False
    assert req.metadata["risk_classifier"]["auto_generatable"] is False
    assert req.metadata["risk_classifier"]["reason"] == "blocked_integration_or_private_data"
    assert req.risk_level == RiskLevel.L0
