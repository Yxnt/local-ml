"""Tests for the sandbox verification pipeline (ToolVerifier + sandbox_runner.py).

The full pipeline is: ToolVerifier.verify() does AST static scan, then calls
sandbox_runner.py in a subprocess for import/structure/schema testing.
These tests exercise the end-to-end pipeline.

For direct sandbox_runner.py tests, see the TestDirectSandboxRunner class.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from server.tools.verifier import ToolVerifier

_SANDBOX_RUNNER = Path(__file__).resolve().parent.parent.parent / "server" / "tools" / "sandbox_runner.py"


def _run_sandbox_direct(config: dict, timeout: int = 20) -> dict:
    """Run sandbox_runner.py directly with the given config and return parsed JSON."""
    proc = subprocess.run(
        [sys.executable, str(_SANDBOX_RUNNER)],
        input=json.dumps(config),
        capture_output=True,
        timeout=timeout,
        text=True,
    )
    assert proc.stdout.strip(), f"No stdout from sandbox_runner. stderr:\n{proc.stderr}"
    return json.loads(proc.stdout.strip())


def _make_sandbox(tmp_path) -> str:
    """Create a temporary sandbox directory and return its string path."""
    d = tmp_path / "sandbox"
    d.mkdir()
    return str(d)


# ---------------------------------------------------------------------------
# Source fixtures
# ---------------------------------------------------------------------------

VALID_TOOL_SOURCE = textwrap.dedent("""\
    from pydantic import BaseModel

    class InputModel(BaseModel):
        name: str = "world"

    class OutputModel(BaseModel):
        greeting: str

    def run(input: InputModel) -> OutputModel:
        return OutputModel(greeting=f"Hello, {input.name}!")
""")

FORBIDDEN_IMPORT_SOURCE = textwrap.dedent("""\
    import os
    from pydantic import BaseModel

    class InputModel(BaseModel):
        name: str = "world"

    class OutputModel(BaseModel):
        result: str

    def run(input: InputModel) -> OutputModel:
        return OutputModel(result=os.getcwd())
""")

EVAL_SOURCE = textwrap.dedent("""\
    from pydantic import BaseModel

    class InputModel(BaseModel):
        expr: str = "1+1"

    class OutputModel(BaseModel):
        result: str

    def run(input: InputModel) -> OutputModel:
        return OutputModel(result=str(eval(input.expr)))
""")


# ---------------------------------------------------------------------------
# Full pipeline tests (ToolVerifier -> sandbox_runner.py)
# ---------------------------------------------------------------------------


class TestSandboxRunnerValidTool:
    """sandbox_runner.py with valid tool code returns passed=true."""

    def test_verify_valid_tool(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        result = verifier.verify("hello_tool", VALID_TOOL_SOURCE)
        assert result.passed is True
        assert len(result.errors) == 0

    def test_execute_valid_tool_directly(self, tmp_path):
        """Direct sandbox_runner execution with arguments."""
        sandbox_dir = _make_sandbox(tmp_path)

        config = {
            "tool_name": "hello_tool",
            "source_code": VALID_TOOL_SOURCE,
            "arguments": {"name": "Bob"},
            "mode": "execute",
            "sandbox_dir": sandbox_dir,
            "timeout": 15,
        }
        output = _run_sandbox_direct(config)
        assert output["success"] is True
        assert output["result"]["greeting"] == "Hello, Bob!"


class TestSandboxRunnerForbiddenImport:
    """sandbox_runner.py with forbidden import returns passed=false.

    This tests the full pipeline: ToolVerifier catches forbidden imports
    in the static scan (AST analysis) before sandbox_runner is invoked.
    """

    def test_forbidden_import_returns_passed_false(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        result = verifier.verify("os_tool", FORBIDDEN_IMPORT_SOURCE)
        assert result.passed is False
        assert any("os" in e for e in result.errors)


class TestSandboxRunnerEval:
    """sandbox_runner.py with eval returns passed=false.

    This tests the full pipeline: ToolVerifier catches eval() calls
    in the static scan (AST analysis) before sandbox_runner is invoked.
    """

    def test_eval_returns_passed_false(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)
        verifier = ToolVerifier(sandbox_dir=sandbox_dir)
        result = verifier.verify("eval_tool", EVAL_SOURCE)
        assert result.passed is False
        assert any("eval" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Direct sandbox_runner.py tests (what it catches on its own)
# ---------------------------------------------------------------------------


class TestDirectSandboxRunner:
    """Direct sandbox_runner.py invocation tests.

    sandbox_runner.py in verify mode checks:
    1. Module imports successfully
    2. Has InputModel, OutputModel, run
    3. BaseModel subclass check
    4. run() signature (exactly 1 parameter)
    5. Schema test (instantiate + call run)
    """

    def test_valid_tool_passes(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)

        config = {
            "tool_name": "hello_tool",
            "source_code": VALID_TOOL_SOURCE,
            "test_input": None,
            "mode": "verify",
            "sandbox_dir": sandbox_dir,
            "timeout": 15,
        }
        output = _run_sandbox_direct(config)
        assert output["passed"] is True
        assert output["errors"] == []

    def test_missing_input_model_fails(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)

        source = textwrap.dedent("""\
            from pydantic import BaseModel

            class OutputModel(BaseModel):
                result: str

            def run(input):
                return OutputModel(result="ok")
        """)
        config = {
            "tool_name": "no_input",
            "source_code": source,
            "test_input": None,
            "mode": "verify",
            "sandbox_dir": sandbox_dir,
            "timeout": 15,
        }
        output = _run_sandbox_direct(config)
        assert output["passed"] is False
        assert any("InputModel" in e for e in output["errors"])

    def test_wrong_return_type_fails(self, tmp_path):
        sandbox_dir = _make_sandbox(tmp_path)

        source = textwrap.dedent("""\
            from pydantic import BaseModel

            class InputModel(BaseModel):
                name: str = "world"

            class OutputModel(BaseModel):
                result: str

            def run(input: InputModel):
                return {"result": "not an OutputModel"}
        """)
        config = {
            "tool_name": "bad_return",
            "source_code": source,
            "test_input": None,
            "mode": "verify",
            "sandbox_dir": sandbox_dir,
            "timeout": 15,
        }
        output = _run_sandbox_direct(config)
        assert output["passed"] is False
        assert any("OutputModel" in e for e in output["errors"])

    def test_run_raising_is_warning(self, tmp_path):
        """run() raising on dummy input produces a warning, not an error."""
        sandbox_dir = _make_sandbox(tmp_path)

        source = textwrap.dedent("""\
            from pydantic import BaseModel

            class InputModel(BaseModel):
                name: str = "world"

            class OutputModel(BaseModel):
                result: str

            def run(input: InputModel) -> OutputModel:
                raise RuntimeError("always fails")
        """)
        config = {
            "tool_name": "crash_tool",
            "source_code": source,
            "test_input": None,
            "mode": "verify",
            "sandbox_dir": sandbox_dir,
            "timeout": 15,
        }
        output = _run_sandbox_direct(config)
        assert len(output["warnings"]) > 0
        assert any("always fails" in w for w in output["warnings"])
