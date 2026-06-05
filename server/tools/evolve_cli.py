"""CLI entry point for the tool evolution pipeline.

Usage::

    python -m server.tools.evolve_cli <command> [options]

Commands:
    requests    Process pending tool requests from telemetry
    absorber    Run the tool absorber (find and merge similar tools)
    metrics     Show EGL and other evolution metrics
    promote     Promote CANDIDATE tools to ACTIVE
    run-once    Run all evolution steps in sequence

All output is JSON to stdout for machine readability.
Warnings and errors are logged to stderr.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Logging: warnings/errors to stderr, never pollute stdout.
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("evolve_cli")


# ---------------------------------------------------------------------------
# JSON output helper
# ---------------------------------------------------------------------------


def dump_json(obj: Any) -> None:
    """Print *obj* as JSON to stdout."""
    print(json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Component factories
# ---------------------------------------------------------------------------


def make_registry(db_path: str) -> Any:
    from server.tools.registry import ToolRegistry

    registry = ToolRegistry(db_path=db_path)
    registry.connect()
    return registry


def make_telemetry(db_path: str) -> Any:
    from server.tools.telemetry import TelemetryService

    telemetry = TelemetryService(db_path=db_path)
    telemetry.connect()
    return telemetry


def make_developer(sandbox_dir: str) -> Any:
    from server.tools.developer import ToolDeveloper

    return ToolDeveloper(sandbox_dir=sandbox_dir)


def make_verifier(sandbox_dir: str) -> Any:
    from server.tools.verifier import ToolVerifier

    return ToolVerifier(sandbox_dir=sandbox_dir)


def make_absorber(
    registry: Any, telemetry: Any, verifier: Any, sandbox_dir: str
) -> Any:
    from server.tools.absorber import ToolAbsorber

    return ToolAbsorber(
        registry=registry,
        telemetry=telemetry,
        verifier=verifier,
        sandbox_dir=sandbox_dir,
    )


def make_retriever(registry: Any) -> Any:
    from server.tools.retriever import ToolRetriever

    return ToolRetriever(registry=registry)


def make_orchestrator(db_path: str, sandbox_dir: str) -> tuple[Any, Any, Any]:
    """Build and return (orchestrator, registry, telemetry).

    The caller is responsible for disconnecting registry and telemetry.
    """
    from server.tools.orchestrator import ToolEvolutionOrchestrator

    telemetry = make_telemetry(db_path)
    registry = make_registry(db_path)
    developer = make_developer(sandbox_dir)
    verifier = make_verifier(sandbox_dir)
    absorber = make_absorber(registry, telemetry, verifier, sandbox_dir)
    retriever = make_retriever(registry)

    orchestrator = ToolEvolutionOrchestrator(
        registry=registry,
        telemetry=telemetry,
        developer=developer,
        verifier=verifier,
        absorber=absorber,
        retriever=retriever,
    )
    return orchestrator, registry, telemetry


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def cmd_requests(args: argparse.Namespace) -> None:
    """Process pending tool requests."""
    orchestrator, registry, telemetry = make_orchestrator(
        args.db_path, args.sandbox_dir
    )
    try:
        result = asyncio.run(
            orchestrator.process_pending_requests(
                limit=args.limit, dry_run=args.dry_run
            )
        )
        dump_json(result)
    except Exception as exc:
        logger.error("requests command failed: %s", exc)
        dump_json({"error": str(exc)})
        sys.exit(1)
    finally:
        registry.disconnect()
        telemetry.disconnect()


def cmd_absorber(args: argparse.Namespace) -> None:
    """Run the absorber."""
    orchestrator, registry, telemetry = make_orchestrator(
        args.db_path, args.sandbox_dir
    )
    try:
        result = asyncio.run(
            orchestrator.run_absorber(dry_run=args.dry_run)
        )
        dump_json(result)
    except Exception as exc:
        logger.error("absorber command failed: %s", exc)
        dump_json({"error": str(exc)})
        sys.exit(1)
    finally:
        registry.disconnect()
        telemetry.disconnect()


def cmd_metrics(args: argparse.Namespace) -> None:
    """Show evolution metrics."""
    orchestrator, registry, telemetry = make_orchestrator(
        args.db_path, args.sandbox_dir
    )
    try:
        result = orchestrator.compute_metrics()
        dump_json(result)
    except Exception as exc:
        logger.error("metrics command failed: %s", exc)
        dump_json({"error": str(exc)})
        sys.exit(1)
    finally:
        registry.disconnect()
        telemetry.disconnect()


def cmd_promote(args: argparse.Namespace) -> None:
    """Promote CANDIDATE tools to ACTIVE."""
    orchestrator, registry, telemetry = make_orchestrator(
        args.db_path, args.sandbox_dir
    )
    try:
        result = orchestrator.promote_candidates(
            min_success_count=args.min_success_count,
            min_success_rate=args.min_success_rate,
        )
        dump_json(result)
    except Exception as exc:
        logger.error("promote command failed: %s", exc)
        dump_json({"error": str(exc)})
        sys.exit(1)
    finally:
        registry.disconnect()
        telemetry.disconnect()


def cmd_run_once(args: argparse.Namespace) -> None:
    """Run all evolution steps."""
    orchestrator, registry, telemetry = make_orchestrator(
        args.db_path, args.sandbox_dir
    )
    try:
        result = asyncio.run(orchestrator.run_once(dry_run=args.dry_run))
        dump_json(result)
    except Exception as exc:
        logger.error("run-once command failed: %s", exc)
        dump_json({"error": str(exc)})
        sys.exit(1)
    finally:
        registry.disconnect()
        telemetry.disconnect()


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evolve_cli",
        description="Tool evolution pipeline CLI — process requests, "
        "run absorber, promote candidates, show metrics.",
    )
    parser.add_argument(
        "--db-path",
        default="memory/usage.db",
        help="Path to the SQLite database (default: memory/usage.db)",
    )
    parser.add_argument(
        "--sandbox-dir",
        default="server/tools/sandbox",
        help="Directory for generated tool files (default: server/tools/sandbox)",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- requests --
    p_requests = sub.add_parser(
        "requests", help="Process pending tool requests from telemetry"
    )
    p_requests.add_argument(
        "--limit", type=int, default=5, help="Max requests to process (default: 5)"
    )
    dry_apply_group(p_requests)
    p_requests.set_defaults(func=cmd_requests)

    # -- absorber --
    p_absorber = sub.add_parser(
        "absorber", help="Run the tool absorber (find and merge similar tools)"
    )
    dry_apply_group(p_absorber)
    p_absorber.set_defaults(func=cmd_absorber)

    # -- metrics --
    p_metrics = sub.add_parser("metrics", help="Show evolution metrics")
    p_metrics.set_defaults(func=cmd_metrics)

    # -- promote --
    p_promote = sub.add_parser(
        "promote", help="Promote CANDIDATE tools to ACTIVE"
    )
    p_promote.add_argument(
        "--min-success-count",
        type=int,
        default=3,
        help="Minimum success count for promotion (default: 3)",
    )
    p_promote.add_argument(
        "--min-success-rate",
        type=float,
        default=0.8,
        help="Minimum success rate for promotion (default: 0.8)",
    )
    p_promote.set_defaults(func=cmd_promote)

    # -- run-once --
    p_run = sub.add_parser(
        "run-once", help="Run all evolution steps in sequence"
    )
    dry_apply_group(p_run)
    p_run.set_defaults(func=cmd_run_once)

    return parser


def dry_apply_group(parser: argparse.ArgumentParser) -> None:
    """Add mutually exclusive --dry-run / --apply flags."""
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run in dry-run mode, no side effects (default)",
    )
    group.add_argument(
        "--apply",
        action="store_false",
        dest="dry_run",
        help="Apply changes for real (disable dry-run)",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


# Support ``python -m server.tools.evolve_cli``.
if __name__ == "__main__":
    main()
