"""Darwinian Evolver Adapter — optional bridge to an external evolution CLI.

This adapter exports candidate tools + synthetic tests to a temporary git repo,
calls an external CLI command, reads back evolved code, re-verifies it, and
registers it as a new version if valid.

Key constraints:
    * Default DISABLED (``tool_evolution.darwinian.enabled: false``).
    * Does NOT import any ``darwinian_evolver`` Python package.
    * Does NOT pass real user data — only synthetic test data derived from
      the tool's ``input_schema``.
    * Gracefully skips if the CLI is not found on ``PATH``.
    * Only processes tools with ``risk_level`` in (L0, L1) and
      ``status == CANDIDATE``.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server.tools.registry import ToolRegistry
from server.tools.spec import RiskLevel, ToolSpec, ToolStatus
from server.tools.telemetry import TelemetryService
from server.tools.verifier import ToolVerifier

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Allowed risk levels for evolution
# ---------------------------------------------------------------------------

_ALLOWED_RISK_LEVELS: frozenset[RiskLevel] = frozenset({RiskLevel.L0, RiskLevel.L1})

# ---------------------------------------------------------------------------
# Environment variable filter
# ---------------------------------------------------------------------------

_SENSITIVE_KEYWORDS = frozenset({"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"})


def _sanitized_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with sensitive variables removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(kw in k.upper() for kw in _SENSITIVE_KEYWORDS)
    }


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EvolverConfig:
    """Configuration for the Darwinian Evolver adapter.

    All fields have safe defaults — the adapter is disabled by default.
    """

    enabled: bool = False
    cli_command: str = "uv run darwinian_evolver"
    timeout_sec: int = 300
    max_iterations: int = 3
    work_dir: str = "/tmp/local_ml_evolver"


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class DarwinianEvolverAdapter:
    """Thin wrapper that bridges the tool evolution pipeline to an external
    Darwinian Evolver CLI.

    The adapter is **disabled by default**.  When disabled or when the CLI is
    not available, every public method returns an empty list / no-op result.

    Args:
        config: Evolver configuration (loaded from YAML or defaults).
        registry: Central tool registry.
        verifier: Validates generated code before registration.
        telemetry: Event logging service.
    """

    def __init__(
        self,
        config: EvolverConfig,
        registry: ToolRegistry,
        verifier: ToolVerifier,
        telemetry: TelemetryService,
    ) -> None:
        self._config = config
        self._registry = registry
        self._verifier = verifier
        self._telemetry = telemetry

    # -- public API ----------------------------------------------------------

    async def evolve_candidates(self, limit: int = 3) -> list[dict[str, Any]]:
        """Try to evolve *limit* CANDIDATE tools via the external evolver.

        Returns a list of result dicts, one per candidate::

            {
                "tool_name": str,
                "status": "evolved" | "skipped" | "failed",
                "details": str,
            }

        If the adapter is disabled or the CLI is missing the method returns
        an empty list after logging the reason.
        """
        if not self._config.enabled:
            logger.info("DarwinianEvolverAdapter: disabled, skipping")
            return []

        if not self._check_cli_available():
            logger.warning(
                "DarwinianEvolverAdapter: CLI not found: %s", self._config.cli_command
            )
            return []

        # Fetch eligible candidates
        candidates = self._eligible_candidates(limit)
        if not candidates:
            logger.info("DarwinianEvolverAdapter: no eligible candidates")
            return []

        results: list[dict[str, Any]] = []

        for spec in candidates:
            result = await self._evolve_one(spec)
            results.append(result)

        return results

    # -- eligibility ---------------------------------------------------------

    def _eligible_candidates(self, limit: int) -> list[ToolSpec]:
        """Return up to *limit* CANDIDATE tools with allowed risk levels."""
        all_tools = self._registry.list_tools(status=ToolStatus.CANDIDATE)
        eligible = [
            t
            for t in all_tools
            if t.risk_level in _ALLOWED_RISK_LEVELS
        ]
        return eligible[:limit]

    # -- single-tool evolution -----------------------------------------------

    async def _evolve_one(self, spec: ToolSpec) -> dict[str, Any]:
        """Run the full evolution pipeline for a single tool spec."""
        tool_name = spec.name
        logger.info("DarwinianEvolverAdapter: evolving %s", tool_name)

        # 1. Export candidate to temp work dir
        try:
            export = self._export_candidate(spec)
        except Exception as exc:
            logger.error("Failed to export candidate %s: %s", tool_name, exc)
            self._telemetry.record(
                "tool_evolution_failed",
                tool_name=tool_name,
                error_type="export_error",
                error_message=str(exc)[:500],
            )
            return {"tool_name": tool_name, "status": "failed", "details": f"export error: {exc}"}

        # 2. Run evolver CLI
        evolver_result = self._run_evolver(export["work_dir"])
        if not evolver_result["success"]:
            details = evolver_result.get("error", "evolver returned non-zero")
            logger.warning("Evolver failed for %s: %s", tool_name, details)
            self._telemetry.record(
                "tool_evolution_failed",
                tool_name=tool_name,
                error_type="evolver_error",
                error_message=details[:500],
            )
            return {"tool_name": tool_name, "status": "failed", "details": details}

        # 3. Read evolved code
        evolved_code = self._read_evolved(export["work_dir"], tool_name)
        if evolved_code is None:
            details = "no evolved code found in output"
            logger.warning("Evolver produced no output for %s", tool_name)
            self._telemetry.record(
                "tool_evolution_failed",
                tool_name=tool_name,
                error_type="no_output",
                error_message=details,
            )
            return {"tool_name": tool_name, "status": "failed", "details": details}

        # 4. Verify evolved code
        verify_result = self._verifier.verify(tool_name, evolved_code)
        if not verify_result.passed:
            details = f"verification failed: {verify_result.errors}"
            logger.warning("Evolved code failed verification for %s: %s", tool_name, verify_result.errors)
            self._telemetry.record(
                "tool_evolution_failed",
                tool_name=tool_name,
                error_type="verify_failed",
                error_message="; ".join(verify_result.errors)[:500],
            )
            return {"tool_name": tool_name, "status": "failed", "details": details}

        # 5. Register as new version
        new_version = self._bump_version(spec.version)
        evolved_spec = ToolSpec(
            name=spec.name,
            description=spec.description,
            input_schema=spec.input_schema,
            output_schema=spec.output_schema,
            runtime=spec.runtime,
            provider=spec.provider,
            entrypoint=spec.entrypoint,
            risk_level=spec.risk_level,
            privacy_scope=spec.privacy_scope,
            version=new_version,
            status=ToolStatus.CANDIDATE,
            tags=spec.tags + ["darwinian_evolved"],
            metadata={
                **spec.metadata,
                "evolved_by": "darwinian_evolver",
                "evolved_from_version": spec.version,
                "evolved_at": datetime.now(timezone.utc).isoformat(),
                "evolver_score": evolver_result.get("score"),
            },
        )
        self._registry.register(evolved_spec)

        # Record creation telemetry for EGL tracking
        self._telemetry.record_tool_created(
            tool_name,
            new_version,
            metadata={
                "runtime": "python_generated",
                "provider": "evolved",
                "source": "darwinian",
                "risk_level": spec.risk_level.value,
            },
        )

        # Record telemetry
        self._telemetry.record(
            "tool_evolved",
            tool_name=tool_name,
            tool_version=new_version,
            metadata={
                "previous_version": spec.version,
                "evolver_score": evolver_result.get("score"),
            },
        )

        logger.info(
            "DarwinianEvolverAdapter: evolved %s %s -> %s",
            tool_name,
            spec.version,
            new_version,
        )
        return {
            "tool_name": tool_name,
            "status": "evolved",
            "details": f"registered as v{new_version}",
        }

    # -- export candidate ----------------------------------------------------

    def _export_candidate(self, spec: ToolSpec) -> dict[str, Any]:
        """Export tool code + synthetic test to a temporary directory.

        The directory is initialised as a git repo so the evolver CLI can
        treat it as a standalone project.

        Returns::

            {
                "tool_name": str,
                "work_dir": str,
                "code_path": str,
                "test_path": str,
                "spec_path": str,
            }
        """
        work_dir = Path(self._config.work_dir) / spec.name
        work_dir.mkdir(parents=True, exist_ok=True)

        # Write spec JSON
        spec_path = work_dir / "spec.json"
        spec_path.write_text(
            json.dumps(spec.to_db_row(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        # Write existing source code if available in sandbox
        sandbox_dir = Path("server/tools/sandbox")
        existing_source = sandbox_dir / f"{spec.name}.py"
        code_path = work_dir / f"{spec.name}.py"
        if existing_source.is_file():
            code_path.write_text(existing_source.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            # Write a stub so the evolver has something to start from
            code_path.write_text(
                self._generate_stub(spec),
                encoding="utf-8",
            )

        # Generate and write synthetic test
        test_code = self._generate_synthetic_test(spec)
        test_path = work_dir / f"test_{spec.name}.py"
        test_path.write_text(test_code, encoding="utf-8")

        # Initialise git repo (ignore errors — git may not be configured)
        try:
            subprocess.run(
                ["git", "init"],
                cwd=str(work_dir),
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "add", "-A"],
                cwd=str(work_dir),
                capture_output=True,
                timeout=10,
            )
            subprocess.run(
                ["git", "commit", "-m", f"export {spec.name} for evolution"],
                cwd=str(work_dir),
                capture_output=True,
                timeout=10,
            )
        except Exception as exc:
            logger.debug("Git init failed (non-fatal): %s", exc)

        logger.info("Exported candidate %s to %s", spec.name, work_dir)
        return {
            "tool_name": spec.name,
            "work_dir": str(work_dir),
            "code_path": str(code_path),
            "test_path": str(test_path),
            "spec_path": str(spec_path),
        }

    # -- run evolver ---------------------------------------------------------

    def _run_evolver(self, work_dir: str) -> dict[str, Any]:
        """Call the external Darwinian Evolver CLI.

        The CLI is invoked with:
            <cli_command> --work-dir <work_dir> --max-iterations <N>

        Environment is sanitized to remove sensitive variables.

        Returns::

            {
                "success": bool,
                "best_code_path": str | None,
                "score": float | None,
                "error": str | None,
            }
        """
        parts = self._config.cli_command.split()
        cmd = parts + [
            "--work-dir", work_dir,
            "--max-iterations", str(self._config.max_iterations),
        ]

        env = _sanitized_env()

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._config.timeout_sec,
                env=env,
                cwd=work_dir,
            )
        except FileNotFoundError:
            return {
                "success": False,
                "best_code_path": None,
                "score": None,
                "error": f"CLI not found: {parts[0]}",
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "best_code_path": None,
                "score": None,
                "error": f"CLI timed out after {self._config.timeout_sec}s",
            }
        except Exception as exc:
            return {
                "success": False,
                "best_code_path": None,
                "score": None,
                "error": f"Failed to run CLI: {exc}",
            }

        if proc.returncode != 0:
            stderr_snippet = (proc.stderr or "").strip()[-500:]
            return {
                "success": False,
                "best_code_path": None,
                "score": None,
                "error": f"CLI exited with code {proc.returncode}: {stderr_snippet}",
            }

        # Try to parse JSON output from stdout
        return self._parse_evolver_output(proc.stdout, work_dir)

    def _parse_evolver_output(self, stdout: str, work_dir: str) -> dict[str, Any]:
        """Parse the evolver CLI's JSON output.

        Expected format::

            {"success": true, "best_code_path": "...", "score": 0.95}

        Falls back to searching for ``evolved_*.py`` in *work_dir* if the
        output is not valid JSON.
        """
        try:
            data = json.loads(stdout.strip())
            return {
                "success": bool(data.get("success", False)),
                "best_code_path": data.get("best_code_path"),
                "score": data.get("score"),
                "error": data.get("error"),
            }
        except (json.JSONDecodeError, TypeError):
            pass

        # Fallback: look for evolved files in the work directory
        evolved_dir = Path(work_dir) / "output"
        if evolved_dir.is_dir():
            py_files = sorted(evolved_dir.glob("*.py"))
            if py_files:
                return {
                    "success": True,
                    "best_code_path": str(py_files[-1]),
                    "score": None,
                    "error": None,
                }

        # Last resort: look for any evolved_*.py in work_dir
        evolved_files = sorted(Path(work_dir).glob("evolved_*.py"))
        if evolved_files:
            return {
                "success": True,
                "best_code_path": str(evolved_files[-1]),
                "score": None,
                "error": None,
            }

        return {
            "success": False,
            "best_code_path": None,
            "score": None,
            "error": f"Could not parse CLI output or find evolved code: {stdout[:300]}",
        }

    # -- read evolved code ---------------------------------------------------

    def _read_evolved(self, work_dir: str, tool_name: str) -> str | None:
        """Read the evolved source code from the evolver's output.

        Checks several conventional output locations in order:
        1. ``<work_dir>/output/<tool_name>.py``
        2. ``<work_dir>/evolved_<tool_name>.py``
        3. ``<work_dir>/<tool_name>.py`` (in-place modification)

        Returns the source code string, or ``None`` if no evolved file is found.
        """
        candidates = [
            Path(work_dir) / "output" / f"{tool_name}.py",
            Path(work_dir) / f"evolved_{tool_name}.py",
            Path(work_dir) / f"{tool_name}.py",
        ]

        for path in candidates:
            if path.is_file():
                code = path.read_text(encoding="utf-8")
                if code.strip():
                    logger.info("Read evolved code from %s", path)
                    return code

        return None

    # -- synthetic test generation -------------------------------------------

    def _generate_synthetic_test(self, spec: ToolSpec) -> str:
        """Generate a synthetic test file for the tool.

        Test data is derived **solely** from ``spec.input_schema`` — no real
        user data is ever used.  The generated test imports the tool module
        and exercises ``run()`` with synthetic ``InputModel`` instances.
        """
        tool_module = spec.name
        schema = spec.input_schema
        properties: dict[str, Any] = schema.get("properties", {})
        required: list[str] = schema.get("required", [])

        # Build synthetic values for each property
        assignments: list[str] = []
        for prop_name, prop_def in properties.items():
            value = self._synthetic_value(prop_name, prop_def)
            assignments.append(f"        {prop_name}={value},")

        if assignments:
            input_block = "\n".join(assignments)
            input_instantiation = textwrap.dedent(f"""\
                input_data = InputModel(
{input_block}
                )""")
        else:
            input_instantiation = "input_data = InputModel()"

        # Determine the import path — the tool module sits in sandbox/
        test_code = textwrap.dedent(f"""\
            \"\"\"Synthetic tests for {spec.name}.

            Auto-generated by DarwinianEvolverAdapter.
            All test data is synthetic — derived from input_schema only.
            \"\"\"

            import sys
            from pathlib import Path

            # Ensure the tool module is importable
            sandbox_dir = Path(__file__).resolve().parent
            if str(sandbox_dir) not in sys.path:
                sys.path.insert(0, str(sandbox_dir))

            try:
                from {tool_module} import InputModel, OutputModel, run
            except ImportError as e:
                print(f"SKIP: cannot import {{e}}")
                sys.exit(0)


            def test_input_model_instantiation():
                \"\"\"Verify InputModel can be constructed with synthetic data.\"\"\"
                {input_instantiation}
                assert input_data is not None


            def test_run_returns_output_model():
                \"\"\"Verify run() returns an OutputModel instance.\"\"\"
                {input_instantiation}
                result = run(input_data)
                assert isinstance(result, OutputModel), (
                    f"Expected OutputModel, got {{type(result)}}"
                )


            def test_run_does_not_raise():
                \"\"\"Verify run() completes without raising.\"\"\"
                {input_instantiation}
                try:
                    run(input_data)
                except Exception as e:
                    assert False, f"run() raised {{type(e).__name__}}: {{e}}"


            if __name__ == "__main__":
                test_input_model_instantiation()
                test_run_returns_output_model()
                test_run_does_not_raise()
                print("ALL SYNTHETIC TESTS PASSED")
        """)

        return test_code

    def _synthetic_value(self, prop_name: str, prop_def: dict[str, Any]) -> str:
        """Return a Python source literal that is a plausible synthetic value
        for a schema property.

        Only uses the property name and JSON Schema definition — never real
        user data.
        """
        prop_type = prop_def.get("type", "string")
        default = prop_def.get("default")
        enum_values = prop_def.get("enum")
        examples = prop_def.get("examples")

        # Use enum or examples if available (they came from the schema, not users)
        if enum_values:
            return repr(enum_values[0])
        if examples:
            return repr(examples[0])
        if default is not None:
            return repr(default)

        # Derive from type
        if prop_type == "string":
            # Use the property name to make a meaningful synthetic string
            return repr(f"test_{prop_name}")
        if prop_type == "integer":
            return "1"
        if prop_type == "number":
            return "1.0"
        if prop_type == "boolean":
            return "True"
        if prop_type == "array":
            return "[]"
        if prop_type == "object":
            return "{}"

        # Fallback
        return repr(f"test_{prop_name}")

    # -- helpers -------------------------------------------------------------

    def _generate_stub(self, spec: ToolSpec) -> str:
        """Generate a minimal stub module for a tool that has no existing code."""
        properties: dict[str, Any] = spec.input_schema.get("properties", {})
        required: list[str] = spec.input_schema.get("required", [])

        # Build InputModel fields
        fields: list[str] = []
        for prop_name, prop_def in properties.items():
            py_type = self._json_type_to_python(prop_def.get("type", "string"))
            is_optional = prop_name not in required
            if is_optional:
                fields.append(f"    {prop_name}: {py_type} | None = None")
            else:
                fields.append(f"    {prop_name}: {py_type}")

        if fields:
            fields_block = "\n".join(fields)
        else:
            fields_block = "    pass"

        # Build OutputModel
        output_schema = spec.output_schema or {}
        output_properties: dict[str, Any] = output_schema.get("properties", {})
        output_fields: list[str] = []
        for prop_name, prop_def in output_properties.items():
            py_type = self._json_type_to_python(prop_def.get("type", "string"))
            output_fields.append(f"    {prop_name}: {py_type} | None = None")

        if output_fields:
            output_block = "\n".join(output_fields)
        else:
            output_block = "    result: str = ''"

        stub = textwrap.dedent(f"""\
            \"\"\"Auto-generated stub for {spec.name}.
            {spec.description}
            \"\"\"

            from pydantic import BaseModel


            class InputModel(BaseModel):
            {fields_block}


            class OutputModel(BaseModel):
            {output_block}


            def run(input: InputModel) -> OutputModel:
                # TODO: implement {spec.name}
                return OutputModel()
        """)

        return stub

    @staticmethod
    def _json_type_to_python(json_type: str) -> str:
        """Map a JSON Schema type to a Python type annotation string."""
        mapping = {
            "string": "str",
            "integer": "int",
            "number": "float",
            "boolean": "bool",
            "array": "list",
            "object": "dict",
        }
        return mapping.get(json_type, "Any")

    @staticmethod
    def _bump_version(version: str) -> str:
        """Increment the patch component of a semver string.

        ``"1.0.0"`` -> ``"1.0.1"``, ``"2.3.9"`` -> ``"2.3.10"``.
        If the version is not valid semver, appends ``".1"``.
        """
        parts = version.split(".")
        try:
            parts[-1] = str(int(parts[-1]) + 1)
            return ".".join(parts)
        except (ValueError, IndexError):
            return f"{version}.1"

    def _check_cli_available(self) -> bool:
        """Check whether the CLI command is available on PATH.

        Uses ``shutil.which`` on the first token of ``config.cli_command``
        (e.g. ``"uv"`` from ``"uv run darwinian_evolver"``).
        """
        first_token = self._config.cli_command.split()[0]
        return shutil.which(first_token) is not None
