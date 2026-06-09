"""End-to-end test: similar tools -> merge -> register -> deprecate.

Covers:
  1. Register two similar generated tools with similar embeddings
  2. Fake aggregator returns merge=true
  3. Fake merger returns valid merged Python code
  4. run_absorber(dry_run=True) -- verify NO changes to registry
  5. run_absorber(dry_run=False) -- verify merged tool registered,
     parent tools deprecated, tool_lineage written
  6. Merger returning empty code -> must fail, not register
  7. Verifier failure -> parents NOT deprecated
"""

from __future__ import annotations

import json
import math
import textwrap
from typing import Any

import pytest

from server.tools.absorber import ToolAbsorber
from server.tools.spec import (
    RiskLevel,
    ToolRuntime,
    ToolSpec,
    ToolStatus,
)
from server.tools.registry import ToolRegistry
from server.tools.telemetry import TelemetryService


# ---------------------------------------------------------------------------
# Two similar tool source files
# ---------------------------------------------------------------------------

TOOL_A_SOURCE = textwrap.dedent('''\
    """Count characters in text."""

    from pydantic import BaseModel


    class InputModel(BaseModel):
        text: str

    class OutputModel(BaseModel):
        char_count: int

    def run(input: InputModel) -> OutputModel:
        return OutputModel(char_count=len(input.text))
''')

TOOL_B_SOURCE = textwrap.dedent('''\
    """Count characters excluding spaces."""

    from pydantic import BaseModel


    class InputModel(BaseModel):
        text: str

    class OutputModel(BaseModel):
        char_count_no_spaces: int

    def run(input: InputModel) -> OutputModel:
        return OutputModel(char_count_no_spaces=len(input.text.replace(" ", "")))
''')

MERGED_SOURCE = textwrap.dedent('''\
    """Merged character counting tool.

    # merged: char_counter_merged
    """

    from pydantic import BaseModel


    class InputModel(BaseModel):
        text: str
        mode: str = "all"

    class OutputModel(BaseModel):
        char_count: int = 0
        char_count_no_spaces: int = 0

    def run(input: InputModel) -> OutputModel:
        total = len(input.text)
        no_spaces = len(input.text.replace(" ", ""))
        return OutputModel(char_count=total, char_count_no_spaces=no_spaces)
''')

BROKEN_MERGED_SOURCE = textwrap.dedent('''\
    """Broken merged character counting tool.

    # merged: char_counter_merged
    """

    from pydantic import BaseModel


    class InputModel(BaseModel):
        text: str

    class OutputModel(BaseModel):
        char_count: int = 0
        char_count_no_spaces: int = 0

    def run(input: InputModel) -> OutputModel:
        return OutputModel(char_count=0, char_count_no_spaces=0)
''')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_embedding(base: list[float], noise_seed: int = 0) -> list[float]:
    """Create a slightly perturbed copy of base embedding.

    The perturbation is small enough that cosine similarity stays above 0.86.
    """
    result = []
    for i, v in enumerate(base):
        noise = 0.01 * ((noise_seed + i) % 5 - 2)  # small noise in [-0.02, +0.02]
        result.append(v + noise)
    # Normalize
    norm = math.sqrt(sum(x * x for x in result))
    return [x / norm for x in result]


def _base_embedding(dim: int = 768) -> list[float]:
    """Generate a base embedding vector."""
    import random
    rng = random.Random(42)
    vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec]


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Fake remote_generate
# ---------------------------------------------------------------------------

def _make_fake_remote_generate(merge_decision: bool = True, merge_code: str = MERGED_SOURCE):
    """Return an async callable that acts as both aggregator and merger."""

    async def _remote_generate(system_prompt: str, user_prompt: str) -> str:
        if "aggregator" in system_prompt.lower() or "decide" in system_prompt.lower():
            reason = "Similar character counting tools" if merge_decision else "Different functionality"
            return json.dumps({"merge": merge_decision, "reason": reason})
        # Merger prompt
        return merge_code

    return _remote_generate


# ---------------------------------------------------------------------------
# Fake Verifier
# ---------------------------------------------------------------------------

class FakeVerifier:
    """A verifier that passes or fails based on configuration.

    Always rejects empty or whitespace-only source code.
    """

    def __init__(self, should_pass: bool = True) -> None:
        self._should_pass = should_pass
        self.verify_calls: list[tuple[str, str]] = []

    def verify(self, tool_name: str, source_code: str):
        self.verify_calls.append((tool_name, source_code))

        class _Result:
            def __init__(self, passed: bool, errors: list[str]):
                self.passed = passed
                self.errors = errors

        # Always reject empty code
        if not source_code or not source_code.strip():
            return _Result(False, ["Empty source code"])

        if self._should_pass:
            return _Result(True, [])
        return _Result(False, ["Fake verification failure"])


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def telemetry(tmp_path):
    svc = TelemetryService(db_path=str(tmp_path / "telemetry.db"))
    svc.connect()
    yield svc
    svc.disconnect()


@pytest.fixture()
def registry(tmp_path, telemetry):
    db_path = str(tmp_path / "registry.db")
    reg = ToolRegistry(db_path=db_path, telemetry=telemetry)
    reg.connect()
    yield reg
    reg.disconnect()


@pytest.fixture()
def sandbox_dir(tmp_path):
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


def _register_two_similar_tools(
    registry: ToolRegistry,
    sandbox_dir: str,
    embedding_dim: int = 768,
    replay_cases_a: list[dict[str, Any]] | None = None,
    replay_cases_b: list[dict[str, Any]] | None = None,
) -> tuple[ToolSpec, ToolSpec]:
    """Register two similar tools with similar embeddings. Returns (spec_a, spec_b)."""
    # Write source files to sandbox
    with open(f"{sandbox_dir}/char_counter_a.py", "w") as f:
        f.write(TOOL_A_SOURCE)
    with open(f"{sandbox_dir}/char_counter_b.py", "w") as f:
        f.write(TOOL_B_SOURCE)

    base_emb = _base_embedding(embedding_dim)
    emb_a = _make_embedding(base_emb, noise_seed=0)
    emb_b = _make_embedding(base_emb, noise_seed=1)

    # Verify embeddings are actually similar
    assert _cosine_sim(emb_a, emb_b) > 0.86, (
        f"Test embeddings too dissimilar: {_cosine_sim(emb_a, emb_b):.4f}"
    )

    input_schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }
    output_schema = {
        "type": "object",
        "properties": {"count": {"type": "integer"}},
    }

    spec_a = ToolSpec(
        name="char_counter_a",
        description="Count characters in text",
        input_schema=input_schema,
        output_schema=output_schema,
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L0,
        status=ToolStatus.CANDIDATE,
        embedding=emb_a,
    )
    spec_b = ToolSpec(
        name="char_counter_b",
        description="Count characters excluding spaces in text",
        input_schema=input_schema,
        output_schema=output_schema,
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        risk_level=RiskLevel.L0,
        status=ToolStatus.CANDIDATE,
        embedding=emb_b,
    )

    if replay_cases_a:
        spec_a.metadata["replay_cases"] = replay_cases_a
    if replay_cases_b:
        spec_b.metadata["replay_cases"] = replay_cases_b

    registry.register(spec_a)
    registry.register(spec_b)
    return spec_a, spec_b


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_no_changes(registry, telemetry, sandbox_dir):
    """run_absorber(dry_run=True) should NOT modify the registry."""
    spec_a, spec_b = _register_two_similar_tools(registry, sandbox_dir)
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(merge_decision=True),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=True)

    assert result["clusters_found"] >= 1
    assert result["clusters_merged"] == 0  # dry run -- no actual merges
    assert result["tools_deprecated"] == 0

    # Both original tools still exist and are not deprecated
    a = registry.get_tool("char_counter_a")
    b = registry.get_tool("char_counter_b")
    assert a is not None
    assert b is not None
    assert a.status == ToolStatus.CANDIDATE
    assert b.status == ToolStatus.CANDIDATE


@pytest.mark.asyncio
async def test_full_merge_and_deprecate(registry, telemetry, sandbox_dir):
    """run_absorber(dry_run=False) merges and deprecates parents."""
    spec_a, spec_b = _register_two_similar_tools(registry, sandbox_dir)
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(
            merge_decision=True, merge_code=MERGED_SOURCE
        ),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_found"] >= 1
    assert result["clusters_merged"] >= 1
    assert result["tools_deprecated"] >= 2

    # Merged tool registered
    merged = registry.get_tool("char_counter_merged")
    assert merged is not None
    assert merged.status == ToolStatus.ACTIVE

    # Parents deprecated
    a = registry.get_tool("char_counter_a")
    b = registry.get_tool("char_counter_b")
    # get_tool returns the first available version; check via list_tools
    all_tools = registry.list_tools(status=None)
    deprecated_names = {
        t.name for t in all_tools if t.status == ToolStatus.DEPRECATED
    }
    assert "char_counter_a" in deprecated_names
    assert "char_counter_b" in deprecated_names

    # Lineage written
    lineage = absorber.get_lineage("char_counter_merged")
    assert len(lineage) >= 2
    parent_names = {r["parent_tool_name"] for r in lineage}
    assert "char_counter_a" in parent_names
    assert "char_counter_b" in parent_names

    # Telemetry recorded
    events = telemetry.get_recent_events(limit=20)
    event_types = [e["event_type"] for e in events]
    assert "tool_merged" in event_types


@pytest.mark.asyncio
async def test_absorber_replay_cases_pass_for_compatible_merge(registry, telemetry, sandbox_dir):
    """Replay-compatible merges preserve parent behavior contracts."""
    _register_two_similar_tools(
        registry,
        sandbox_dir,
        replay_cases_a=[
            {
                "name": "counts_ascii_chars",
                "arguments": {"text": "abc"},
                "expect": {"char_count": 3},
                "match": "subset",
            }
        ],
        replay_cases_b=[
            {
                "name": "counts_non_space_chars",
                "arguments": {"text": "a b"},
                "expect": {"char_count_no_spaces": 2},
                "match": "subset",
            }
        ],
    )
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(
            merge_decision=True,
            merge_code=MERGED_SOURCE,
        ),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_merged"] == 1
    assert registry.get_tool("char_counter_merged") is not None


@pytest.mark.asyncio
async def test_absorber_replay_cases_block_incompatible_merge(registry, telemetry, sandbox_dir):
    """Behavior-breaking merged tools are rejected even when structure verifies."""
    _register_two_similar_tools(
        registry,
        sandbox_dir,
        replay_cases_a=[
            {
                "name": "protects_character_count",
                "arguments": {"text": "abc"},
                "expect": {"char_count": 3},
                "match": "subset",
            }
        ],
        replay_cases_b=[
            {
                "name": "protects_non_space_count",
                "arguments": {"text": "a b"},
                "expect": {"char_count_no_spaces": 2},
                "match": "subset",
            }
        ],
    )
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(
            merge_decision=True,
            merge_code=BROKEN_MERGED_SOURCE,
        ),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_merged"] == 0
    assert any(detail["action"] == "parent_tests_failed" for detail in result["details"])
    assert registry.get_tool("char_counter_merged") is None


@pytest.mark.asyncio
async def test_merger_empty_code_fails(registry, telemetry, sandbox_dir):
    """Merger returning empty code -> merge fails, parents NOT deprecated."""
    spec_a, spec_b = _register_two_similar_tools(registry, sandbox_dir)
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(
            merge_decision=True, merge_code=""
        ),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    # Cluster found but merge failed
    assert result["clusters_found"] >= 1
    # With empty source code, either merge fails or verify fails
    # In either case, no clusters should be merged
    # Check details for failure
    details = result["details"]
    for d in details:
        if d.get("cluster"):
            assert d.get("merged") is False or d.get("action") in (
                "merge_failed", "verify_failed", "parent_tests_failed"
            )

    # Parents NOT deprecated
    all_tools = registry.list_tools(status=None)
    deprecated_names = {
        t.name for t in all_tools if t.status == ToolStatus.DEPRECATED
    }
    assert "char_counter_a" not in deprecated_names
    assert "char_counter_b" not in deprecated_names


@pytest.mark.asyncio
async def test_verifier_failure_no_deprecation(registry, telemetry, sandbox_dir):
    """Verifier failure -> parents NOT deprecated."""
    spec_a, spec_b = _register_two_similar_tools(registry, sandbox_dir)
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=False),
        remote_generate=_make_fake_remote_generate(
            merge_decision=True, merge_code=MERGED_SOURCE
        ),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_found"] >= 1
    # Merge attempted but verify failed
    details = result["details"]
    for d in details:
        if d.get("cluster"):
            assert d.get("merged") is False
            assert d.get("action") in ("verify_failed", "parent_tests_failed")

    # Parents NOT deprecated
    all_tools = registry.list_tools(status=None)
    deprecated_names = {
        t.name for t in all_tools if t.status == ToolStatus.DEPRECATED
    }
    assert "char_counter_a" not in deprecated_names
    assert "char_counter_b" not in deprecated_names


@pytest.mark.asyncio
async def test_aggregator_says_no_merge(registry, telemetry, sandbox_dir):
    """When aggregator says merge=false, no merge happens."""
    spec_a, spec_b = _register_two_similar_tools(registry, sandbox_dir)
    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(merge_decision=False),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)

    assert result["clusters_found"] >= 1
    assert result["clusters_merged"] == 0

    # All tools still active/candidate
    all_tools = registry.list_tools(status=None)
    deprecated_names = {
        t.name for t in all_tools if t.status == ToolStatus.DEPRECATED
    }
    assert len(deprecated_names) == 0


@pytest.mark.asyncio
async def test_no_similar_tools_no_clusters(registry, telemetry, sandbox_dir):
    """With dissimilar embeddings, absorber finds no clusters."""
    # Register two tools with very different embeddings
    emb_a = [1.0] + [0.0] * 767
    emb_b = [0.0] * 767 + [1.0]

    spec_a = ToolSpec(
        name="tool_alpha",
        description="Alpha tool",
        input_schema={"type": "object", "properties": {}},
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        status=ToolStatus.CANDIDATE,
        embedding=emb_a,
    )
    spec_b = ToolSpec(
        name="tool_beta",
        description="Beta tool",
        input_schema={"type": "object", "properties": {}},
        runtime=ToolRuntime.PYTHON_GENERATED,
        provider="generated",
        status=ToolStatus.CANDIDATE,
        embedding=emb_b,
    )
    registry.register(spec_a)
    registry.register(spec_b)

    absorber = ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=FakeVerifier(should_pass=True),
        remote_generate=_make_fake_remote_generate(),
        sandbox_dir=sandbox_dir,
    )

    result = await absorber.run(dry_run=False)
    assert result["clusters_found"] == 0
    assert result["clusters_merged"] == 0
