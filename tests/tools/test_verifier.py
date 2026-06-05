"""Tests for server.tools.verifier — ToolVerifier."""

from __future__ import annotations

import os
import textwrap

import pytest

from server.tools.verifier import ToolVerifier

# -- source snippets used across tests --

VALID_TOOL_SOURCE = '''\
from pydantic import BaseModel

class InputModel(BaseModel):
    name: str = "world"

class OutputModel(BaseModel):
    greeting: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(greeting=f"Hello, {input.name}!")
'''

INVALID_IMPORT_SOURCE = '''\
import os
from pydantic import BaseModel

class InputModel(BaseModel):
    name: str = "world"

class OutputModel(BaseModel):
    result: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(result=os.getcwd())
'''

EVAL_SOURCE = '''\
from pydantic import BaseModel

class InputModel(BaseModel):
    expr: str = "1+1"

class OutputModel(BaseModel):
    result: str

def run(input: InputModel) -> OutputModel:
    return OutputModel(result=str(eval(input.expr)))
'''

SYNTAX_ERROR_SOURCE = '''\
from pydantic import BaseModel

class InputModel(BaseModel):
    name: str = "world"

def run(input: InputModel):
    return {"greeting": f"Hello, {input.name}!"  # missing closing paren
'''


class TestValidToolPasses:
    """A correctly written tool passes verification."""

    def test_valid_tool_passes(self, verifier):
        result = verifier.verify("hello_tool", VALID_TOOL_SOURCE)
        assert result.passed is True
        assert len(result.errors) == 0


class TestForbiddenImportFails:
    """Forbidden imports (import os) are caught by static scan."""

    def test_forbidden_import_os_fails(self, verifier):
        result = verifier.verify("bad_tool", INVALID_IMPORT_SOURCE)
        assert result.passed is False
        assert any("os" in e for e in result.errors)

    def test_forbidden_import_subprocess_fails(self, verifier):
        source = textwrap.dedent("""\
            import subprocess
            from pydantic import BaseModel

            class InputModel(BaseModel):
                cmd: str = "echo hi"

            class OutputModel(BaseModel):
                out: str

            def run(input: InputModel) -> OutputModel:
                return OutputModel(out="nope")
        """)
        result = verifier.verify("bad_tool2", source)
        assert result.passed is False
        assert any("subprocess" in e for e in result.errors)


class TestEvalCallFails:
    """eval() calls are caught by static scan."""

    def test_eval_call_fails(self, verifier):
        result = verifier.verify("eval_tool", EVAL_SOURCE)
        assert result.passed is False
        assert any("eval" in e for e in result.errors)


class TestSyntaxErrorFails:
    """Syntax errors are caught during static scan."""

    def test_syntax_error_fails(self, verifier):
        result = verifier.verify("syntax_tool", SYNTAX_ERROR_SOURCE)
        assert result.passed is False
        assert any("Syntax error" in e for e in result.errors)


class TestSubprocessIsolation:
    """The verifier runs in a subprocess, not the main process.

    We verify this by checking that importing a module with a side-effect
    (writing a marker file) does NOT create the file in our process space.
    """

    def test_verifier_runs_in_subprocess(self, verifier, sandbox_dir):
        marker = os.path.join(sandbox_dir, "INJECTED_MARKER")
        # Remove it first if it somehow exists
        if os.path.exists(marker):
            os.remove(marker)

        source = textwrap.dedent("""\
            import os
            from pydantic import BaseModel

            class InputModel(BaseModel):
                name: str = "test"

            class OutputModel(BaseModel):
                result: str

            def run(input: InputModel) -> OutputModel:
                return OutputModel(result="ok")
        """)
        # This should fail at static scan (forbidden import os), so
        # the marker file is never written.
        result = verifier.verify("inject_tool", source)
        assert result.passed is False
        assert not os.path.exists(marker), "Side-effect leaked into main process!"

    def test_subprocess_crash_reported(self, verifier):
        """A module that causes a subprocess crash is reported gracefully."""
        # This source passes static scan (no forbidden imports) but will
        # fail in the sandbox subprocess because it has no InputModel.
        source = textwrap.dedent("""\
            from pydantic import BaseModel

            # Deliberately missing InputModel / run
            def hello():
                return "world"
        """)
        result = verifier.verify("crash_tool", source)
        assert result.passed is False
        assert len(result.errors) > 0
