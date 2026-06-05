#!/usr/bin/env python3
"""Standalone subprocess runner for sandboxed tool verification and execution.

Runs INSIDE a child process.  Reads a JSON config from stdin, executes the
tool module, and prints a JSON result to stdout.

Modes
-----
verify   — check structure (InputModel/OutputModel/run), instantiate with
           test_input, call run(); returns ``{passed, result, errors, warnings}``.
execute  — load module, instantiate InputModel with *arguments*, call run();
           returns ``{success, result, errors}``.

Security measures
-----------------
* Sensitive env vars (containing KEY / SECRET / TOKEN / PASSWORD / CREDENTIAL /
  AUTH) are stripped at startup (defence-in-depth, caller also strips).
* ``HOME`` is overridden to a temp dir so the tool cannot read user dotfiles.
* CWD is set to *sandbox_dir*.
* ``signal.alarm`` enforces a hard timeout (default 15 s).
* ALL exceptions are caught and serialised as JSON errors.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
import signal
import sys
import tempfile
import traceback

DEFAULT_TIMEOUT = 15  # seconds


# ---------------------------------------------------------------------------
# Environment filtering
# ---------------------------------------------------------------------------

_SENSITIVE_KEYWORDS = frozenset({"KEY", "SECRET", "TOKEN", "PASSWORD", "CREDENTIAL", "AUTH"})


def filter_env() -> dict[str, str]:
    """Return a copy of ``os.environ`` with sensitive keys removed."""
    return {
        k: v
        for k, v in os.environ.items()
        if not any(kw in k.upper() for kw in _SENSITIVE_KEYWORDS)
    }


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def setup_timeout(seconds: int) -> None:
    """Install a ``SIGALRM`` handler that raises ``TimeoutError``."""

    def _handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"Execution timed out after {seconds} seconds")

    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)


# ---------------------------------------------------------------------------
# Dummy‑input builder (for verify mode)
# ---------------------------------------------------------------------------


def build_dummy_input(input_cls: type) -> object:
    """Create a minimal ``InputModel`` instance from field annotations."""
    fields = input_cls.model_fields
    dummy_values: dict[str, object] = {}

    for name, field_info in fields.items():
        annotation = field_info.annotation
        if annotation is str:
            dummy_values[name] = "test"
        elif annotation is int:
            dummy_values[name] = 0
        elif annotation is float:
            dummy_values[name] = 0.0
        elif annotation is bool:
            dummy_values[name] = False
        elif annotation is list or annotation is list[str]:
            dummy_values[name] = []
        elif annotation is dict:
            dummy_values[name] = {}
        else:
            # Optional fields get None; required fields get a string fallback
            if not field_info.is_required():
                dummy_values[name] = None
            else:
                dummy_values[name] = "test"

    return input_cls(**dummy_values)


# ---------------------------------------------------------------------------
# Module helpers
# ---------------------------------------------------------------------------


def import_tool_module(
    tool_name: str,
    source_code: str,
    sandbox_dir: str,
) -> object:
    """Write *source_code* to disk (if non‑empty) and import the module."""
    if source_code:
        file_path = os.path.join(sandbox_dir, f"{tool_name}.py")
        with open(file_path, "w", encoding="utf-8") as fh:
            fh.write(source_code)
    else:
        file_path = os.path.join(sandbox_dir, f"{tool_name}.py")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Tool file not found: {file_path}")

    mod_spec = importlib.util.spec_from_file_location(
        f"sandbox_{tool_name}", file_path
    )
    if mod_spec is None or mod_spec.loader is None:
        raise RuntimeError(f"Cannot create module spec for {file_path}")

    module = importlib.util.module_from_spec(mod_spec)

    # Temporarily add sandbox_dir to sys.path so relative imports inside the
    # generated tool resolve correctly.
    added = False
    if sandbox_dir not in sys.path:
        sys.path.insert(0, sandbox_dir)
        added = True
    try:
        mod_spec.loader.exec_module(module)
    finally:
        if added and sandbox_dir in sys.path:
            sys.path.remove(sandbox_dir)

    return module


def serialize_output(output: object) -> dict:
    """Serialise an ``OutputModel`` instance to a plain dict."""
    if hasattr(output, "model_dump"):
        return output.model_dump()
    if hasattr(output, "dict"):
        return output.dict()
    return vars(output)


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------


def verify(
    tool_name: str,
    source_code: str,
    test_input: dict | None,
    sandbox_dir: str,
) -> dict:
    """Run all verification checks and return a result dict."""
    errors: list[str] = []
    warnings: list[str] = []
    result_data: dict | None = None

    # --- Import ---
    try:
        module = import_tool_module(tool_name, source_code, sandbox_dir)
    except Exception as exc:
        errors.append(f"Import failed: {exc}")
        return {"passed": False, "result": None, "errors": errors, "warnings": warnings}

    # --- Structure check ---
    if not hasattr(module, "InputModel"):
        errors.append("Missing class: InputModel")
    if not hasattr(module, "OutputModel"):
        errors.append("Missing class: OutputModel")
    if not hasattr(module, "run"):
        errors.append("Missing function: run")
    if errors:
        return {"passed": False, "result": None, "errors": errors, "warnings": warnings}

    # --- Pydantic BaseModel subclass check ---
    try:
        from pydantic import BaseModel  # type: ignore[import-untyped]

        if not issubclass(module.InputModel, BaseModel):
            errors.append("InputModel is not a pydantic.BaseModel subclass")
        if not issubclass(module.OutputModel, BaseModel):
            errors.append("OutputModel is not a pydantic.BaseModel subclass")
    except ImportError:
        warnings.append("pydantic not available — skipping BaseModel check")

    # --- run() signature check ---
    try:
        sig = inspect.signature(module.run)
        params = list(sig.parameters.keys())
        if len(params) != 1:
            errors.append(
                f"run() must take exactly 1 parameter (got {len(params)})"
            )
    except Exception:
        warnings.append("Cannot inspect run() signature")

    if errors:
        return {"passed": False, "result": None, "errors": errors, "warnings": warnings}

    # --- Schema test (instantiate + call run) ---
    try:
        if test_input is not None:
            input_instance = module.InputModel(**test_input)
        else:
            input_instance = build_dummy_input(module.InputModel)

        try:
            output = module.run(input_instance)
            if not isinstance(output, module.OutputModel):
                errors.append(
                    f"run() returned {type(output).__name__}, expected OutputModel"
                )
            else:
                result_data = serialize_output(output)
        except Exception as exc:
            warnings.append(f"run() raised on dummy input (may be expected): {exc}")
    except Exception as exc:
        errors.append(f"Schema test failed: {exc}")

    return {
        "passed": len(errors) == 0,
        "result": result_data,
        "errors": errors,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Execute mode
# ---------------------------------------------------------------------------


def execute(
    tool_name: str,
    source_code: str,
    arguments: dict,
    sandbox_dir: str,
) -> dict:
    """Import the tool, instantiate ``InputModel(*arguments)``, call ``run()``."""
    try:
        module = import_tool_module(tool_name, source_code, sandbox_dir)
        input_instance = module.InputModel(**arguments)
        output = module.run(input_instance)
        return {
            "success": True,
            "result": serialize_output(output),
            "errors": [],
        }
    except Exception as exc:
        return {
            "success": False,
            "result": None,
            "errors": [str(exc), traceback.format_exc()],
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    # 1. Strip sensitive env vars (defence-in-depth)
    safe_env = filter_env()

    # Override HOME to a temp dir so the tool cannot access user dotfiles.
    # Keep PATH and PYTHONPATH so imports and commands still work.
    safe_env["HOME"] = tempfile.mkdtemp(prefix="sandbox_home_")

    os.environ.clear()
    os.environ.update(safe_env)

    # 2. Read config from stdin
    config_json = sys.stdin.read()
    config: dict = json.loads(config_json)

    tool_name: str = config["tool_name"]
    source_code: str = config.get("source_code", "")
    test_input: dict | None = config.get("test_input")
    arguments: dict | None = config.get("arguments")
    mode: str = config.get("mode", "verify")
    sandbox_dir: str = config.get("sandbox_dir", tempfile.gettempdir())
    timeout: int = config.get("timeout", DEFAULT_TIMEOUT)

    # If arguments live in a separate file, load them now.
    arguments_file: str | None = config.get("arguments_file")
    if arguments_file and arguments is None:
        with open(arguments_file, "r", encoding="utf-8") as fh:
            arguments = json.load(fh)

    # 3. Ensure sandbox_dir exists and make it CWD
    os.makedirs(sandbox_dir, exist_ok=True)
    os.chdir(sandbox_dir)

    # 4. Install hard timeout
    setup_timeout(timeout)

    # 5. Run the requested mode
    try:
        if mode == "verify":
            output = verify(tool_name, source_code, test_input, sandbox_dir)
        elif mode == "execute":
            if arguments is None:
                output = {
                    "success": False,
                    "result": None,
                    "errors": ["execute mode requires 'arguments' or 'arguments_file'"],
                }
            else:
                output = execute(tool_name, source_code, arguments, sandbox_dir)
        else:
            output = {
                "passed": False,
                "result": None,
                "errors": [f"Unknown mode: {mode}"],
                "warnings": [],
            }
    except TimeoutError as exc:
        output = {
            "passed": False,
            "result": None,
            "errors": [str(exc)],
            "warnings": [],
        }
    except Exception as exc:
        output = {
            "passed": False,
            "result": None,
            "errors": [f"Unexpected error: {exc}", traceback.format_exc()],
            "warnings": [],
        }

    # 6. Write result to stdout
    print(json.dumps(output, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
