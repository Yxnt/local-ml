# Local Computer Use Design

## Goal

Enable Gemma 4 and MiniCPM5 to control the macOS desktop through screenshots and mouse/keyboard actions, inspired by OpenAI's computer use API, pi-computer-use, and open-computer-use.

## Architecture

```
computer_use/
  __init__.py       # Public API exports
  screenshot.py     # Screen capture (macOS screencapture)
  actions.py        # Mouse/keyboard control (Quartz/CoreGraphics)
  tools.py          # Tool schema + system prompt
  agent.py          # Main loop: screenshot → model → actions → repeat

computer_use_demo.py  # CLI demo script
```

## Design Decisions

### Why not use pi-computer-use's AX approach?

pi-computer-use uses macOS Accessibility APIs for semantic targeting (@e1, @e2 refs). This is powerful but:
1. Requires native Swift code and macOS permissions (Accessibility, Screen Recording)
2. Small models may not understand AX refs well
3. More complex to implement and maintain

Our approach: coordinate-based actions (like OpenAI's computer use) which is simpler and more universal.

### Why single tool instead of separate tools?

OpenAI uses a single `computer` tool with action types. We follow this pattern:
- Single `computer_action` tool with `action` parameter
- Simpler for small models to understand
- Fewer tool calls needed per step

### Screenshot handling

Small models (Gemma 4, MiniCPM5) have limited vision capabilities. The agent:
1. Takes a screenshot before every action
2. Sends it as base64-encoded PNG in the message
3. Models that support vision can analyze the image
4. Models that don't support vision can still respond based on the task description

## Tool Schema

```json
{
  "type": "function",
  "function": {
    "name": "computer_action",
    "parameters": {
      "type": "object",
      "properties": {
        "action": {
          "type": "string",
          "enum": ["screenshot", "click", "double_click", "move", "drag", "scroll", "type", "keypress", "wait"]
        },
        "x": {"type": "integer"},
        "y": {"type": "integer"},
        "end_x": {"type": "integer"},
        "end_y": {"type": "integer"},
        "text": {"type": "string"},
        "keys": {"type": "array", "items": {"type": "string"}},
        "scroll_x": {"type": "integer"},
        "scroll_y": {"type": "integer"},
        "button": {"type": "string", "enum": ["left", "right"]},
        "seconds": {"type": "number"}
      },
      "required": ["action"]
    }
  }
}
```

## Supported Actions

| Action | Parameters | Description |
|--------|-----------|-------------|
| screenshot | (none) | Capture current screen |
| click | x, y, button | Click at coordinates |
| double_click | x, y | Double-click at coordinates |
| move | x, y | Move mouse cursor |
| drag | x, y, end_x, end_y | Drag from start to end |
| scroll | x, y, scroll_x, scroll_y | Scroll at position |
| type | text | Type text at cursor |
| keypress | keys | Press key combination |
| wait | seconds | Wait for duration |

## Agent Loop

```
1. Take screenshot → base64
2. Send to model with task description
3. Model returns computer_action tool_call
4. Execute the action
5. Go to 1 (until max_steps or model stops)
```

## Known Limitations

1. **Vision support**: Gemma 4 is a VLM but may struggle with precise coordinate detection. MiniCPM5 has limited vision capabilities.
2. **Coordinate accuracy**: Small models may not accurately identify UI element positions from screenshots.
3. **macOS only**: Uses Quartz/CoreGraphics, not cross-platform.
4. **No AX integration**: Doesn't use Accessibility APIs for semantic targeting.

## Usage

```python
from model.backends import ModelRegistry
from computer_use.agent import ComputerUseAgent

registry = ModelRegistry()
registry.register_defaults()
backend = await registry.get_or_load("gemma-4-e2b-it-4bit")

agent = ComputerUseAgent(backend, max_steps=10)
result = agent.run("Take a screenshot of the desktop")
```

## Future Improvements

1. **AX integration**: Add pi-computer-use style accessibility targeting for better accuracy
2. **Grounding model**: Use a specialized grounding model (like OS-Atlas) for coordinate detection
3. **Action history**: Track action history for better context
4. **Error recovery**: Detect and recover from failed actions
5. **Multi-monitor support**: Handle multiple displays
