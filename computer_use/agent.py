"""Computer use agent loop: screenshot -> model -> actions -> repeat."""

import json
import base64
import time
from pathlib import Path
from typing import Any

from computer_use.screenshot import capture_screen_base64
from computer_use import actions
from computer_use.tools import COMPUTER_USE_TOOL, COMPUTER_USE_SYSTEM_PROMPT


class ComputerUseAgent:
    """Agent that lets a VLM model control the computer.

    Usage:
        agent = ComputerUseAgent(backend)
        agent.run("Open Safari and go to example.com")
    """

    def __init__(self, backend: Any, max_steps: int = 20, verbose: bool = True):
        """Initialize the agent.

        Args:
            backend: A ModelBackend instance (from model/backends.py).
            max_steps: Maximum number of action steps before stopping.
            verbose: Whether to print debug info.
        """
        self.backend = backend
        self.max_steps = max_steps
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[CUA] {msg}")

    def _execute_action(self, action_name: str, params: dict[str, Any]) -> None:
        """Execute a single computer action."""
        if action_name == "screenshot":
            return  # Screenshot is handled in the loop

        elif action_name == "click":
            x = params.get("x", 0)
            y = params.get("y", 0)
            button = params.get("button", "left")
            self._log(f"click({x}, {y}, button={button})")
            actions.click(x, y, button)

        elif action_name == "double_click":
            x = params.get("x", 0)
            y = params.get("y", 0)
            self._log(f"double_click({x}, {y})")
            actions.double_click(x, y)

        elif action_name == "move":
            x = params.get("x", 0)
            y = params.get("y", 0)
            self._log(f"move_mouse({x}, {y})")
            actions.move_mouse(x, y)

        elif action_name == "drag":
            start_x = params.get("x", 0)
            start_y = params.get("y", 0)
            end_x = params.get("end_x", 0)
            end_y = params.get("end_y", 0)
            self._log(f"drag({start_x}, {start_y}, {end_x}, {end_y})")
            actions.drag(start_x, start_y, end_x, end_y)

        elif action_name == "scroll":
            x = params.get("x", 0)
            y = params.get("y", 0)
            scroll_x = params.get("scroll_x", 0)
            scroll_y = params.get("scroll_y", 0)
            self._log(f"scroll({x}, {y}, scroll_x={scroll_x}, scroll_y={scroll_y})")
            actions.scroll(x, y, scroll_x, scroll_y)

        elif action_name == "type":
            text = params.get("text", "")
            self._log(f"type_text({text!r})")
            actions.type_text(text)

        elif action_name == "keypress":
            keys = params.get("keys", [])
            self._log(f"keypress({keys})")
            actions.keypress(keys)

        elif action_name == "wait":
            seconds = params.get("seconds", 1.0)
            self._log(f"wait({seconds})")
            actions.wait(seconds)

        else:
            self._log(f"Unknown action: {action_name}")

    def run(self, task: str, screenshot_dir: str | None = None) -> str:
        """Run the computer use agent on a task.

        Args:
            task: Natural language task description.
            screenshot_dir: Optional directory to save screenshots for debugging.

        Returns:
            The model's final response text.
        """
        if screenshot_dir:
            Path(screenshot_dir).mkdir(parents=True, exist_ok=True)

        # Build initial messages
        messages = [
            {"role": "system", "content": COMPUTER_USE_SYSTEM_PROMPT},
            {"role": "user", "content": task},
        ]

        tools = [COMPUTER_USE_TOOL]

        for step in range(self.max_steps):
            self._log(f"Step {step + 1}/{self.max_steps}")

            # Take screenshot
            screenshot_b64 = capture_screen_base64()

            if screenshot_dir:
                screenshot_path = Path(screenshot_dir) / f"step_{step:03d}.png"
                screenshot_path.write_bytes(base64.b64decode(screenshot_b64))

            # Add screenshot to messages
            # For models that support vision, we encode the screenshot in the prompt
            # Small models may not support vision well, so we also describe the action
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_b64}",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Here is the current screenshot. What action should I take next? Use the computer_action tool.",
                    },
                ],
            })

            # Build prompt
            prompt = self.backend.apply_chat_template(
                [{"role": m["role"], "content": str(m["content"])} for m in messages],
                tools=tools,
                enable_thinking=False,
            )

            # Generate response
            self._log("Generating response...")
            output_text = self.backend.generate(prompt=prompt, max_tokens=1024)
            self._log(f"Model output: {output_text[:200]}...")

            # Parse tool calls
            tool_calls = self.backend.parse_tool_calls(output_text)

            if not tool_calls:
                # No tool call - model might be done or confused
                self._log("No tool call returned, treating as final response")
                return output_text

            # Execute the first tool call
            tc = tool_calls[0]
            fn_name = tc["function"]["name"]
            fn_args = json.loads(tc["function"]["arguments"])

            if fn_name != "computer_action":
                self._log(f"Unexpected tool call: {fn_name}")
                return output_text

            action_name = fn_args.get("action")
            if not action_name:
                self._log("No action specified in tool call")
                return output_text

            # Execute the action
            self._execute_action(action_name, fn_args)

            # Small delay after action
            time.sleep(0.2)

        return f"Reached maximum steps ({self.max_steps}) without completion."
