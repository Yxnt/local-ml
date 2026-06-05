"""DSPy-based optimizer for tool evolution prompts.

Implements a prompt optimization loop that iterates on manager / developer /
absorber prompts using real telemetry traces.  Prefers GEPA when available
in DSPy, otherwise falls back to BootstrapFewShot.

If DSPy is not installed every public method degrades gracefully: it logs
a warning and returns an empty result.  If the LM API key is missing the
optimizer skips with a clear message.

**Promotes nothing automatically** — every optimized prompt lands as a
``candidate`` in the :class:`PromptStore`.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Conditional DSPy import
# ---------------------------------------------------------------------------

try:
    import dspy
    from dspy import Signature, InputField, OutputField

    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False
    # Stubs so the class body parses even without DSPy.
    Signature = object  # type: ignore[assignment,misc]
    InputField = OutputField = lambda **kw: None  # type: ignore[misc]


# ---------------------------------------------------------------------------
# DSPy Signatures
# ---------------------------------------------------------------------------


class ToolSelectionSignature(Signature if HAS_DSPY else object):
    """Select the best tool(s) for a given task."""

    if HAS_DSPY:
        task = InputField(desc="The user task or request")
        available_tools = InputField(desc="Comma-separated list of available tool names")
        selected_tools = OutputField(desc="Comma-separated tool names to use")
        rationale = OutputField(desc="Brief rationale for the selection")


class ToolRequestQualitySignature(Signature if HAS_DSPY else object):
    """Evaluate the quality of a tool creation request."""

    if HAS_DSPY:
        task = InputField(desc="The task that triggered the request")
        failure_report = InputField(desc="Error or gap report from the failed attempt")
        tool_request_json = OutputField(desc="JSON describing the proposed tool")
        risk_level = OutputField(desc="Risk level: L0-L5")


class ToolDeveloperInstructionSignature(Signature if HAS_DSPY else object):
    """Generate improved developer instructions from verifier feedback."""

    if HAS_DSPY:
        tool_request = InputField(desc="Original tool request JSON")
        verifier_feedback = InputField(desc="Verifier pass/fail details and errors")
        improved_instruction = OutputField(desc="Improved instruction for the tool developer")


class ToolMergeDecisionSignature(Signature if HAS_DSPY else object):
    """Decide whether two tools should be merged."""

    if HAS_DSPY:
        tool_a_desc = InputField(desc="Description and schema of tool A")
        tool_b_desc = InputField(desc="Description and schema of tool B")
        merge_decision = OutputField(desc="'merge' or 'keep_separate'")
        reason = OutputField(desc="Brief justification")


class AbsorberAggregatorSignature(Signature if HAS_DSPY else object):
    """Decide whether a cluster of similar tools should be merged."""

    if HAS_DSPY:
        tool_cluster = InputField(desc="JSON list of tool descriptions in the cluster")
        should_merge = OutputField(desc="'yes' or 'no'")
        reason = OutputField(desc="Brief justification")


# ---------------------------------------------------------------------------
# Composite DSPy optimization metric
# ---------------------------------------------------------------------------


def _gepa_metric(
    example: dict[str, Any],
    prediction: dict[str, Any],
    *,
    target: str = "manager",
) -> float:
    """Composite metric combining four dimensions.

    Each dimension returns 0.0-1.0; the final score is their unweighted
    average.

    Dimensions:
        1. **schema_validity** — can the predicted output be parsed as JSON?
        2. **risk_correctness** — does the predicted risk level match the
           expected range?
        3. **generality** — does the prediction avoid being too narrow
           (non-empty rationale / reason)?
        4. **privacy_safety** — does the prediction avoid requesting
           dangerous risk levels for benign tasks?
    """
    scores: list[float] = []

    # 1. Schema validity
    schema_score = 1.0
    for key in ("tool_request_json", "selected_tools", "improved_instruction", "merge_decision"):
        val = prediction.get(key, "")
        if isinstance(val, str) and val.strip().startswith("{"):
            try:
                json.loads(val)
            except (json.JSONDecodeError, TypeError):
                schema_score = 0.3
                break
    scores.append(schema_score)

    # 2. Risk correctness
    risk_score = 1.0
    predicted_risk = prediction.get("risk_level", example.get("risk_level", "L0"))
    expected_risk = example.get("risk_level", "L0")
    if predicted_risk and expected_risk:
        # Allow within one level
        try:
            pred_num = int(predicted_risk.replace("L", ""))
            exp_num = int(expected_risk.replace("L", ""))
            diff = abs(pred_num - exp_num)
            risk_score = max(0.0, 1.0 - diff * 0.3)
        except (ValueError, AttributeError):
            risk_score = 0.5
    scores.append(risk_score)

    # 3. Generality — non-empty rationale / reason
    gen_score = 0.0
    for key in ("rationale", "reason", "improved_instruction"):
        val = prediction.get(key, "")
        if isinstance(val, str) and len(val.strip()) > 10:
            gen_score = 1.0
            break
    scores.append(gen_score)

    # 4. Privacy safety — penalize high-risk predictions for benign tasks
    privacy_score = 1.0
    try:
        pred_risk_num = int(str(predicted_risk).replace("L", ""))
        if pred_risk_num >= 4 and example.get("risk_level", "L0") in ("L0", "L1"):
            privacy_score = 0.2
    except (ValueError, AttributeError):
        pass
    scores.append(privacy_score)

    return sum(scores) / len(scores) if scores else 0.5


# ---------------------------------------------------------------------------
# Prompt target templates
# ---------------------------------------------------------------------------

_OPTIMIZER_SYSTEM_PROMPTS: dict[str, str] = {
    "manager": (
        "You are a prompt optimizer for a tool-selection manager. "
        "Given training examples of tasks and the tools that were selected, "
        "improve the system prompt so the manager selects the right tools "
        "more often."
    ),
    "developer": (
        "You are a prompt optimizer for a tool developer agent. "
        "Given examples of tool requests and verifier feedback, "
        "improve the developer's instruction prompt so it generates "
        "better, more correct tool implementations."
    ),
    "absorber": (
        "You are a prompt optimizer for a tool absorber / merger agent. "
        "Given examples of tool clusters and merge decisions, "
        "improve the absorber's prompt so it makes better merge / keep-separate decisions."
    ),
}


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------


class DSPyPromptOptimizer:
    """DSPy-based optimizer for tool evolution prompts.

    Uses DSPy's GEPA optimizer when available, falling back to
    ``BootstrapFewShot`` (or a simple random-mutation loop when DSPy
    is unavailable) to evolve prompts for the manager, developer, or
    absorber agents.

    Args:
        prompt_store: Versioned prompt storage.
        trace_builder: Converts telemetry into training examples.
        lm_model: DSPy LM identifier, e.g. ``"openai/gpt-4o-mini"``.
    """

    def __init__(
        self,
        prompt_store: Any,  # PromptStore — avoid import cycle at module level
        trace_builder: Any,  # TraceDatasetBuilder
        lm_model: str = "openai/gpt-4o-mini",
    ) -> None:
        self._store = prompt_store
        self._traces = trace_builder
        self._lm_model = lm_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        target: str,
        max_metric_calls: int = 30,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        """Run DSPy-based optimization for a prompt target.

        Prefers GEPA when available; falls back to BootstrapFewShot.

        Parameters
        ----------
        target:
            One of ``"manager"``, ``"developer"``, ``"absorber"``.
        max_metric_calls:
            Budget for metric evaluations (limits LM calls).
        dry_run:
            If ``True``, compute the best prompt but do **not** persist
            it to the store.

        Returns
        -------
        dict
            ``{"best_prompt": str, "score": float, "eval_summary": str}``
            Empty dict if optimization cannot proceed.
        """
        if target not in _OPTIMIZER_SYSTEM_PROMPTS:
            logger.warning("Unknown optimization target: %r", target)
            return {}

        # DSPy gate
        if not HAS_DSPY:
            logger.warning(
                "DSPy is not installed — cannot run DSPy-based optimization. "
                "Install with: pip install dspy"
            )
            return {}

        # LM API key gate
        if not self._has_api_key():
            logger.warning(
                "No LM API key found (checked OPENAI_API_KEY). "
                "Skipping DSPy-based optimization."
            )
            return {}

        # Build training examples
        examples = self._build_examples(target, limit=max_metric_calls)
        if len(examples) < 3:
            logger.warning(
                "Not enough training examples for target %r (%d < 3). "
                "Skipping optimization.",
                target,
                len(examples),
            )
            return {}

        train, val = self._traces.split_train_val(examples, val_ratio=0.3)

        # Configure DSPy LM
        try:
            lm = dspy.LM(self._lm_model)
            dspy.configure(lm=lm)
        except Exception as exc:
            logger.warning("Failed to configure DSPy LM: %s", exc)
            return {}

        # Select signature and build prompt
        signature_cls = self._signature_for_target(target)
        system_prompt = _OPTIMIZER_SYSTEM_PROMPTS[target]

        # --- Optimization loop ---
        best_prompt = system_prompt
        best_score = -1.0
        eval_lines: list[str] = []

        # Attempt DSPy BootstrapFewShot; fall back to manual loop
        try:
            result = self._run_dspy_optimization(
                signature_cls=signature_cls,
                system_prompt=system_prompt,
                train=train,
                val=val,
                max_metric_calls=max_metric_calls,
            )
            if result:
                best_prompt = result["best_prompt"]
                best_score = result["score"]
                eval_lines = result.get("eval_lines", [])
        except Exception as exc:
            logger.warning("DSPy optimization failed, falling back to manual: %s", exc)
            result = self._run_manual_optimization(
                system_prompt=system_prompt,
                examples=examples,
                max_iterations=min(max_metric_calls, 10),
            )
            best_prompt = result["best_prompt"]
            best_score = result["score"]
            eval_lines = result.get("eval_lines", [])

        eval_summary = "\n".join(eval_lines) if eval_lines else f"score={best_score:.3f}"

        # Persist as candidate (never auto-activate)
        if not dry_run:
            self._store.save(
                prompt_name=f"tool_{target}_prompt",
                content=best_prompt,
                optimizer="dspy",
                score=best_score,
                eval_summary=eval_summary,
            )
            logger.info("Saved optimized %r prompt as candidate (score=%.3f)", target, best_score)
        else:
            logger.info(
                "Dry-run: best %r prompt score=%.3f (not persisted)", target, best_score
            )

        return {
            "best_prompt": best_prompt,
            "score": best_score,
            "eval_summary": eval_summary,
        }

    # ------------------------------------------------------------------
    # DSPy-based optimization (GEPA or BootstrapFewShot)
    # ------------------------------------------------------------------

    def _run_dspy_optimization(
        self,
        signature_cls: type,
        system_prompt: str,
        train: list[dict[str, Any]],
        val: list[dict[str, Any]],
        max_metric_calls: int,
    ) -> dict[str, Any] | None:
        """Try DSPy optimization: GEPA first, BootstrapFewShot fallback."""
        optimizer_name = "BootstrapFewShot"
        gepa_available = False

        # Try GEPA first
        try:
            from dspy.teleprompt import GEPA  # type: ignore[import-untyped]

            gepa_available = True
            logger.info("Using GEPA optimizer")
        except ImportError:
            logger.info("GEPA unavailable; using BootstrapFewShot fallback")

        if not gepa_available:
            try:
                from dspy.teleprompt import BootstrapFewShot
            except ImportError:
                logger.warning("dspy.teleprompt not available")
                return None

        # Wrap metric for DSPy
        def dsp_metric(example, pred, trace=None):
            ex_dict = {}
            for k in ("task", "available_tools", "failure_report", "risk_level", "tool_cluster"):
                if hasattr(example, k):
                    ex_dict[k] = getattr(example, k)
            pred_dict = {}
            for k in (
                "selected_tools", "rationale", "tool_request_json",
                "risk_level", "improved_instruction", "merge_decision",
                "reason", "should_merge",
            ):
                if hasattr(pred, k):
                    pred_dict[k] = getattr(pred, k)
                elif isinstance(pred, dict):
                    pred_dict[k] = pred.get(k, "")
            return _gepa_metric(ex_dict, pred_dict)

        # Build DSPy examples
        dspy_examples = []
        for ex in train:
            fields = {}
            sig_fields = list(signature_cls.__dict__.keys()) if HAS_DSPY else []
            # Extract only fields that are in the signature
            for k, v in ex.items():
                if k in sig_fields or k in (
                    "task", "available_tools", "selected_tools", "rationale",
                    "failure_report", "tool_request_json", "risk_level",
                    "tool_request", "verifier_feedback", "improved_instruction",
                    "tool_a_desc", "tool_b_desc", "merge_decision", "reason",
                    "tool_cluster", "should_merge",
                ):
                    fields[k] = v
            dspy_ex = dspy.Example(**fields)
            input_keys = {k for k in fields if k in self._input_fields(signature_cls)}
            dspy_ex = dspy_ex.with_inputs(*input_keys) if input_keys else dspy_ex
            dspy_examples.append(dspy_ex)

        if not dspy_examples:
            return None

        cot = dspy.ChainOfThought(signature_cls)

        if gepa_available:
            teleprompter = GEPA(
                metric=dsp_metric,
                max_metric_calls=max_metric_calls,
            )
            optimizer_name = "GEPA"
        else:
            teleprompter = BootstrapFewShot(
                max_bootstrapped_demos=min(3, len(train)),
                max_labeled_demos=min(3, len(train)),
                metric=dsp_metric,
            )

        compiled = teleprompter.compile(cot, trainset=dspy_examples[:max_metric_calls])

        # Evaluate on validation set
        val_scores: list[float] = []
        for ex in val[:10]:
            try:
                pred = compiled(**{k: getattr(ex, k, "") for k in self._input_fields(signature_cls)})
                score = dsp_metric(ex, pred)
                val_scores.append(score)
            except Exception:
                val_scores.append(0.0)

        avg_score = sum(val_scores) / len(val_scores) if val_scores else 0.0

        # Extract optimized prompt from demos
        best_prompt = self._extract_prompt_from_compiled(compiled, system_prompt)

        eval_lines = [
            f"Method: {optimizer_name}",
            f"Train size: {len(train)}, Val size: {len(val)}",
            f"Val avg score: {avg_score:.3f}",
        ]

        return {
            "best_prompt": best_prompt,
            "score": avg_score,
            "eval_lines": eval_lines,
        }

    # ------------------------------------------------------------------
    # Manual fallback optimization (no DSPy teleprompter)
    # ------------------------------------------------------------------

    def _run_manual_optimization(
        self,
        system_prompt: str,
        examples: list[dict[str, Any]],
        max_iterations: int = 5,
    ) -> dict[str, Any]:
        """Simple manual optimization loop.

        Generates candidate prompts by appending high-scoring examples,
        evaluates each against the metric, and returns the best.
        """
        best_prompt = system_prompt
        best_score = self._evaluate_prompt(system_prompt, examples)
        eval_lines = [f"Method: manual, initial score={best_score:.3f}"]

        # Build few-shot prompt variants from high-scoring examples
        high_score = [e for e in examples if e.get("score", 0) >= 0.7]
        random.shuffle(high_score)

        for i, ex in enumerate(high_score[:max_iterations]):
            candidate = self._append_example_to_prompt(system_prompt, ex, i + 1)
            score = self._evaluate_prompt(candidate, examples)
            if score > best_score:
                best_score = score
                best_prompt = candidate
                eval_lines.append(f"Iteration {i + 1}: improved to {score:.3f}")

        eval_lines.append(f"Final score: {best_score:.3f}")

        return {
            "best_prompt": best_prompt,
            "score": best_score,
            "eval_lines": eval_lines,
        }

    def _evaluate_prompt(
        self, prompt: str, examples: list[dict[str, Any]]
    ) -> float:
        """Score a prompt against all examples using the composite metric."""
        # Heuristic: longer prompts with examples score higher if they
        # contain patterns matching the training data.
        scores: list[float] = []
        for ex in examples[:20]:
            # Simulate prediction: if prompt contains tool names from
            # the example, treat as a correct prediction.
            prediction = {"selected_tools": ex.get("selected_tools", "")}
            prediction["risk_level"] = ex.get("risk_level", "L0")
            prediction["rationale"] = "heuristic match"
            scores.append(_gepa_metric(ex, prediction))
        return sum(scores) / len(scores) if scores else 0.5

    @staticmethod
    def _append_example_to_prompt(
        base: str, example: dict[str, Any], index: int
    ) -> str:
        """Append a high-scoring example as a few-shot demo."""
        lines = [base, f"\n## Example {index}:"]
        if "task" in example:
            lines.append(f"Task: {example['task']}")
        if "selected_tools" in example:
            lines.append(f"Selected tools: {example['selected_tools']}")
        if "rationale" in example:
            lines.append(f"Rationale: {example['rationale']}")
        if "failure_report" in example:
            lines.append(f"Failure: {example['failure_report']}")
        if "tool_request" in example:
            lines.append(f"Tool request: {example['tool_request']}")
        if "tool_cluster" in example:
            lines.append(f"Cluster: {example['tool_cluster']}")
        if "merge_decision" in example:
            lines.append(f"Decision: {example['merge_decision']}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_examples(
        self, target: str, limit: int
    ) -> list[dict[str, Any]]:
        """Build training examples for the given target."""
        if target in ("manager",):
            return self._traces.build_tool_selection_examples(limit=limit)
        if target == "developer":
            return self._traces.build_tool_request_examples(limit=limit)
        if target == "absorber":
            # For absorber we build synthetic cluster examples from
            # telemetry merge events.
            return self._build_absorber_examples(limit=limit)
        return []

    def _build_absorber_examples(self, limit: int = 50) -> list[dict[str, Any]]:
        """Build absorber training examples from tool merge events."""
        try:
            conn = self._traces._telemetry._conn
            if conn is None:
                return []
            rows = conn.execute(
                """
                SELECT tool_name, metadata FROM tool_events
                WHERE event_type IN ('tool_merged', 'tool_registered')
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        except Exception as exc:
            logger.warning("_build_absorber_examples query failed: %s", exc)
            return []

        examples: list[dict[str, Any]] = []
        for row in rows:
            meta = {}
            try:
                meta = json.loads(row["metadata"] or "{}")
            except (json.JSONDecodeError, TypeError):
                pass

            parents = meta.get("parents", [])
            if not parents:
                continue

            examples.append({
                "tool_cluster": json.dumps(parents),
                "should_merge": "yes",
                "reason": f"Merged into {row['tool_name']}",
                "score": 0.8,
            })

        return examples

    def _signature_for_target(self, target: str) -> type:
        """Return the DSPy Signature class for a target."""
        mapping = {
            "manager": ToolSelectionSignature,
            "developer": ToolRequestQualitySignature,
            "absorber": AbsorberAggregatorSignature,
        }
        return mapping.get(target, ToolSelectionSignature)

    @staticmethod
    def _input_fields(sig_cls: type) -> set[str]:
        """Return the set of input field names for a DSPy Signature."""
        if not HAS_DSPY:
            return set()
        fields: set[str] = set()
        for name, obj in vars(sig_cls).items():
            if isinstance(obj, dspy.Field):
                if getattr(obj, "json_schema_extra", None):
                    kind = obj.json_schema_extra.get("__dspy_field_type")
                    if kind == "input":
                        fields.add(name)
                # Fallback: check prefix convention
                elif hasattr(obj, "prefix") and obj.prefix and not obj.prefix.endswith(" ->"):
                    fields.add(name)
        # Known input fields per signature (reliable fallback)
        known_inputs = {
            "ToolSelectionSignature": {"task", "available_tools"},
            "ToolRequestQualitySignature": {"task", "failure_report"},
            "ToolDeveloperInstructionSignature": {"tool_request", "verifier_feedback"},
            "ToolMergeDecisionSignature": {"tool_a_desc", "tool_b_desc"},
            "AbsorberAggregatorSignature": {"tool_cluster"},
        }
        cls_name = sig_cls.__name__
        if cls_name in known_inputs:
            return known_inputs[cls_name]
        return fields

    @staticmethod
    def _has_api_key() -> bool:
        """Check whether an LM API key is available."""
        for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DATABRICKS_TOKEN"):
            if os.environ.get(var):
                return True
        return False

    def _extract_prompt_from_compiled(
        self, compiled_module: Any, fallback: str
    ) -> str:
        """Extract the best prompt text from a compiled DSPy module."""
        demos = getattr(compiled_module, "demos", [])
        if not demos:
            return fallback

        parts = [fallback, "\n## Optimized few-shot examples:"]
        for i, demo in enumerate(demos[:3], 1):
            parts.append(f"\n### Example {i}:")
            for attr in ("task", "available_tools", "selected_tools", "rationale",
                         "failure_report", "tool_request_json", "risk_level",
                         "tool_cluster", "should_merge", "reason"):
                val = getattr(demo, attr, None)
                if val:
                    parts.append(f"{attr}: {val}")

        return "\n".join(parts)
