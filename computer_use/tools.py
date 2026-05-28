"""Tool definitions for computer use."""

COMPUTER_USE_TOOL = {
    "type": "function",
    "function": {
        "name": "computer_action",
        "description": (
            "Perform a computer interaction action. Use this to control the mouse, "
            "keyboard, and interact with the desktop UI. Before clicking or typing, "
            "always take a screenshot first to see the current state."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "screenshot",
                        "click",
                        "double_click",
                        "move",
                        "drag",
                        "scroll",
                        "type",
                        "keypress",
                        "wait",
                    ],
                    "description": "The action to perform.",
                },
                "x": {
                    "type": "integer",
                    "description": "X coordinate for mouse actions (click, double_click, move, drag, scroll).",
                },
                "y": {
                    "type": "integer",
                    "description": "Y coordinate for mouse actions (click, double_click, move, drag, scroll).",
                },
                "end_x": {
                    "type": "integer",
                    "description": "End X coordinate for drag action.",
                },
                "end_y": {
                    "type": "integer",
                    "description": "End Y coordinate for drag action.",
                },
                "text": {
                    "type": "string",
                    "description": "Text to type (for type action).",
                },
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Keys to press simultaneously (for keypress action), e.g. ['cmd', 'c'].",
                },
                "scroll_x": {
                    "type": "integer",
                    "description": "Horizontal scroll amount (positive = right).",
                },
                "scroll_y": {
                    "type": "integer",
                    "description": "Vertical scroll amount (positive = down).",
                },
                "button": {
                    "type": "string",
                    "enum": ["left", "right"],
                    "description": "Mouse button for click action (default: left).",
                },
                "seconds": {
                    "type": "number",
                    "description": "Seconds to wait (for wait action, default: 1.0).",
                },
            },
            "required": ["action"],
        },
    },
}

# System prompt for computer use
COMPUTER_USE_SYSTEM_PROMPT = """You are a computer-use agent. You can see the screen through screenshots and control the computer using mouse and keyboard actions.

IMPORTANT RULES:
1. ALWAYS take a screenshot first before any action to see the current state.
2. Look carefully at the screenshot to understand what's on screen.
3. Identify the exact coordinates of UI elements before clicking.
4. After performing an action, take another screenshot to verify the result.
5. Be precise with coordinates - clicking the wrong place can break the workflow.

Available actions:
- screenshot: Capture the current screen (ALWAYS do this first)
- click: Click at (x, y) coordinates
- double_click: Double-click at (x, y)
- move: Move mouse to (x, y)
- drag: Drag from (x, y) to (end_x, end_y)
- scroll: Scroll at (x, y) with scroll_x/scroll_y amounts
- type: Type text at current cursor position
- keypress: Press key combination (e.g., ["cmd", "c"] to copy)
- wait: Wait for specified seconds

When you complete the task, respond with a summary of what you did."""
