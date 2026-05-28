"""DSPy-based prompt optimizer.

Uses real usage data to automatically improve:
- System prompts
- Tool descriptions
- Response strategies
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

try:
    import dspy
    from dspy import Signature, InputField, OutputField, ChainOfThought
    HAS_DSPY = True
except ImportError:
    HAS_DSPY = False


class ResponseSignature(Signature if HAS_DSPY else object):
    """Signature for agent response generation."""
    if HAS_DSPY:
        user_input = InputField(desc="User's request or question")
        context = InputField(desc="Relevant context (memory, tools, history)")
        system_prompt = InputField(desc="Current system prompt")
        response = OutputField(desc="Best agent response")


class ToolSelectionSignature(Signature if HAS_DSPY else object):
    """Signature for tool selection."""
    if HAS_DSPY:
        user_input = InputField(desc="User's request")
        available_tools = InputField(desc="Available tools with descriptions")
        selected_tool = OutputField(desc="Best tool to use (name only)")


class PromptOptimizer:
    """Optimizes prompts using DSPy.

    Usage:
        optimizer = PromptOptimizer()
        optimizer.configure(lm_model="gpt-4")

        # Load training data
        examples = collector.get_training_examples(min_score=0.5)
        optimizer.load_examples(examples)

        # Optimize system prompt
        best_prompt = optimizer.optimize_system_prompt(current_prompt)
    """

    def __init__(self, lm_model: str = "openai/gpt-4o-mini"):
        if not HAS_DSPY:
            raise ImportError("pip install dspy")

        self._lm_model = lm_model
        self._examples: list[dspy.Example] = []
        self._optimized_prompts: dict[str, str] = {}

    def configure(self, lm_model: str | None = None, api_key: str | None = None) -> None:
        """Configure DSPy settings."""
        if lm_model:
            self._lm_model = lm_model

        lm = dspy.LM(self._lm_model, api_key=api_key)
        dspy.configure(lm=lm)

    def load_examples(self, interactions: list[dict[str, Any]]) -> None:
        """Load training examples from collected interactions."""
        self._examples = []

        for interaction in interactions:
            # Only use successful interactions
            if interaction.get("outcome") not in ("success", "partial"):
                continue

            # Skip if no feedback score or low score
            score = interaction.get("feedback_score")
            if score is not None and score < 0.3:
                continue

            example = dspy.Example(
                user_input=interaction["user_input"],
                context=json.dumps(interaction.get("metadata", {})),
                system_prompt="",  # Will be filled during optimization
                response=interaction["agent_response"],
            ).with_inputs("user_input", "context", "system_prompt")

            self._examples.append(example)

    def optimize_system_prompt(
        self,
        current_prompt: str,
        num_candidates: int = 5,
        max_bootstrapped: int = 3,
    ) -> str:
        """Optimize the system prompt using collected data."""
        if not self._examples:
            raise ValueError("No training examples loaded")

        # Create a ChainOfThought module
        cot = ChainOfThought(ResponseSignature)

        # Bootstrap few-shot examples
        from dspy.teleprompt import BootstrapFewShot

        teleprompter = BootstrapFewShot(
            max_bootstrapped_demos=max_bootstrapped,
            max_labeled_demos=3,
        )

        # Compile with current examples
        compiled = teleprompter.compile(
            cot,
            trainset=self._examples[:20],  # Use subset for speed
        )

        # Extract optimized prompt
        # DSPy optimizes the demos, but we can also extract patterns
        optimized_prompt = self._extract_optimized_prompt(compiled, current_prompt)

        self._optimized_prompts["system"] = optimized_prompt
        return optimized_prompt

    def optimize_tool_descriptions(
        self,
        tools: list[dict[str, Any]],
        usage_stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Optimize tool descriptions based on usage patterns."""
        if not HAS_DSPY:
            return tools

        optimized = []
        for tool in tools:
            tool_copy = dict(tool)
            fn = tool_copy.get("function", {})
            tool_name = fn.get("name", "")

            # Get usage stats for this tool
            stats = usage_stats.get(tool_name, {})
            success_rate = stats.get("success_rate", 0.5)
            usage_count = stats.get("count", 0)

            # If tool has low success rate, improve description
            if success_rate < 0.5 and usage_count > 5:
                improved_desc = self._improve_tool_description(
                    fn.get("description", ""),
                    stats.get("failures", [])
                )
                tool_copy["function"] = {**fn, "description": improved_desc}

            optimized.append(tool_copy)

        return optimized

    def _extract_optimized_prompt(self, compiled_module: Any, current_prompt: str) -> str:
        """Extract optimized prompt from compiled module."""
        # Get the demos that DSPy selected
        demos = getattr(compiled_module, "demos", [])

        if not demos:
            return current_prompt

        # Build optimized prompt with best examples
        parts = [current_prompt]

        # Add successful interaction patterns
        if demos:
            parts.append("\n## 高质量响应示例：")
            for i, demo in enumerate(demos[:3], 1):
                if hasattr(demo, "user_input") and hasattr(demo, "response"):
                    parts.append(f"\n### 示例 {i}:")
                    parts.append(f"用户: {demo.user_input}")
                    parts.append(f"助手: {demo.response}")

        return "\n".join(parts)

    def _improve_tool_description(
        self,
        current_desc: str,
        failures: list[dict[str, Any]],
    ) -> str:
        """Improve tool description based on failure cases."""
        if not failures:
            return current_desc

        # Analyze failures
        failure_reasons = [f.get("reason", "") for f in failures[:5]]

        prompt = f"""当前工具描述: {current_desc}

失败案例:
{chr(10).join(f'- {r}' for r in failure_reasons if r)}

请改进工具描述，使其更清晰准确。只返回新的描述文本。"""

        try:
            lm = dspy.LM(self._lm_model)
            result = lm(prompt)
            return result.strip() if result else current_desc
        except Exception:
            return current_prompt

    def get_optimized_prompt(self, key: str = "system") -> str | None:
        """Get the latest optimized prompt."""
        return self._optimized_prompts.get(key)

    def save_optimizations(self, path: str | Path) -> None:
        """Save optimized prompts to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._optimized_prompts, ensure_ascii=False, indent=2))

    def load_optimizations(self, path: str | Path) -> None:
        """Load optimized prompts from file."""
        path = Path(path)
        if path.exists():
            self._optimized_prompts = json.loads(path.read_text())
