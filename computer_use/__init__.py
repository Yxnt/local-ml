"""Local computer use module for Gemma 4 and MiniCPM5.

Provides screenshot capture, mouse/keyboard control, and an agent loop
that lets small VLM models interact with the macOS desktop.
"""

from computer_use.screenshot import capture_screen, capture_screen_base64
from computer_use.actions import (
    click,
    double_click,
    move_mouse,
    drag,
    scroll,
    type_text,
    keypress,
    wait,
)
from computer_use.tools import COMPUTER_USE_TOOL
from computer_use.agent import ComputerUseAgent

__all__ = [
    "capture_screen",
    "capture_screen_base64",
    "click",
    "double_click",
    "move_mouse",
    "drag",
    "scroll",
    "type_text",
    "keypress",
    "wait",
    "COMPUTER_USE_TOOL",
    "ComputerUseAgent",
]
