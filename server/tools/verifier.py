"""ToolVerifier — validate generated tool code before registration.

Three verification stages:
1. **Static scan** — AST analysis: check structure, block forbidden imports/calls.
2. **Import test** — actually import the module in a subprocess.
3. **Schema test** — instantiate InputModel/OutputModel and run() with dummy data.

Only tools that pass all three stages become ``candidate`` status.

Stages 2-4 run inside a child process (``sandbox_runner.py``) so that buggy
or malicious generated code cannot corrupt the parent process or leak secrets.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from server.tools.spec import ToolSpec, ToolRuntime, ToolStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the subprocess runner
# ---------------------------------------------------------------------------

_SANDBOX_RUNNER = Path(__file__).parent / "sandbox_runner.py"

# ---------------------------------------------------------------------------
# Forbidden names
# ---------------------------------------------------------------------------

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket", "http", "requests",
    "httpx", "ctypes", "signal", "multiprocessing", "threading", "asyncio",
    "importlib", "pkgutil", "zipimport", "compileall", "code", "codeop",
    "compile", "pickle", "shelve", "dbm", "sqlite3", "xmlrpc", "ftplib",
    "smtplib", "imaplib", "poplib", "smtplib", "telnetlib", "ssl",
}

FORBIDDEN_BUILTINS = {
    "eval", "exec", "compile", "__import__", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "breakpoint", "exit", "quit",
}

# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

class VerifyResult:
    """Outcome of tool verification."""

    def __init__(self) -> None:
        self.passed: bool = True
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def fail(self, msg: str) -> None:
        self.passed = False
        self.errors.append(msg)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "errors": self.errors,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Clean environment
# ---------------------------------------------------------------------------

_SENSITIVE_KEYWORDS = frozenset({"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"})


def _clean_env() -> dict[str, str]:
    """Build an env dict that strips sensitive variables."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(kw in k.upper() for kw in _SENSITIVE_KEYWORDS)
    }


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class ToolVerifier:
    """Validate generated Python tool code.

    Args:
        sandbox_dir: Directory containing generated .py files.
    """

    def __init__(self, sandbox_dir: str = "server/tools/sandbox") -> None:
        self._sandbox_dir = Path(sandbox_dir)

    def verify(self, tool_name: str, source_code: str) -> VerifyResult:
        """Run all verification stages on *source_code*.

        Returns a ``VerifyResult`` with ``passed=True`` only if all stages pass.
        """
        result = VerifyResult()

        # -----------------------------------------------------------------
        # Stage 1: Static scan (runs in-process, AST only)
        # -----------------------------------------------------------------
        self._static_scan(source_code, result)
        if not result.passed:
            return result

        # -----------------------------------------------------------------
        # Stages 2-4: import, structure check, schema test (subprocess)
        # -----------------------------------------------------------------
        self._sandbox_dir.mkdir(parents=True, exist_ok=True)

        config = json.dumps(
            {
                "tool_name": tool_name,
                "source_code": source_code,
                "test_input": None,
                "mode": "verify",
                "sandbox_dir": str(self._sandbox_dir.resolve()),
                "timeout": 15,
            },
            ensure_ascii=False,
        )

        try:
            proc = subprocess.run(
                [sys.executable, str(_SANDBOX_RUNNER)],
                input=config,
                capture_output=True,
                timeout=20,
                text=True,
                env=_clean_env(),
            )
        except subprocess.TimeoutExpired:
            result.fail("Sandbox process timed out (20 s)")
            return result
        except Exception as exc:
            result.fail(f"Failed to launch sandbox process: {exc}")
            return result

        # Non-zero exit usually means the runner itself crashed (segfault,
        # unhandled signal, etc.) — the JSON on stdout may still be valid.
        if proc.returncode != 0 and not proc.stdout.strip():
            stderr_snippet = (proc.stderr or "").strip()[-500:]
            result.fail(
                f"Sandbox process exited with code {proc.returncode}: {stderr_snippet}"
            )
            return result

        # Parse JSON output
        try:
            output = json.loads(proc.stdout.strip())
        except json.JSONDecodeError as exc:
            stderr_snippet = (proc.stderr or "").strip()[-500:]
            result.fail(
                f"Invalid JSON from sandbox: {exc}\n"
                f"stdout: {proc.stdout[:300]}\n"
                f"stderr: {stderr_snippet}"
            )
            return result

        # Map sandbox output to VerifyResult
        result.passed = bool(output.get("passed", False))
        result.errors = list(output.get("errors", []))
        result.warnings = list(output.get("warnings", []))

        # If stderr has content, surface it as a warning
        if proc.stderr and proc.stderr.strip():
            result.warn(f"sandbox stderr: {proc.stderr.strip()[-300:]}")

        return result

    # -- Stage 1: Static scan ------------------------------------------------

    def _static_scan(self, source: str, result: VerifyResult) -> None:
        """AST-based static analysis."""
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            result.fail(f"Syntax error: {e}")
            return

        for node in ast.walk(tree):
            # Check imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name.split(".")[0]
                    if mod in FORBIDDEN_MODULES:
                        result.fail(f"Forbidden import: {alias.name}")

            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mod = node.module.split(".")[0]
                    if mod in FORBIDDEN_MODULES:
                        result.fail(f"Forbidden import from: {node.module}")

            # Check function calls to forbidden builtins
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in FORBIDDEN_BUILTINS:
                    result.fail(f"Forbidden call: {func_name}()")

            # Check for file operations outside sandbox
            elif isinstance(node, ast.Call):
                func_name = self._get_call_name(node)
                if func_name in ("open", "Path.open") and not self._is_sandboxed_path(node):
                    result.warn(f"File operation may access non-sandbox path: {func_name}")

        # Check line count
        lines = source.strip().split("\n")
        if len(lines) > 120:
            result.warn(f"Source has {len(lines)} lines (recommended < 80)")

    @staticmethod
    def _get_call_name(node: ast.Call) -> str:
        """Extract the function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        if isinstance(node.func, ast.Attribute):
            return node.func.attr
        return ""

    @staticmethod
    def _is_sandboxed_path(node: ast.Call) -> bool:
        """Heuristic: check if a file path argument contains '/tmp' or 'sandbox'."""
        for arg in node.args:
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                if "/tmp" in arg.value or "sandbox" in arg.value:
                    return True
        return False


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def inspect_signature(func: Any) -> Any:
    """Get function signature, returning None on failure."""
    try:
        import inspect
        return inspect.signature(func)
    except Exception:
        return None
