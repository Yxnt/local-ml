"""Mouse and keyboard control using macOS Quartz/CoreGraphics."""

import time

import Quartz
from Quartz import (
    CGEventCreateMouseEvent,
    CGEventCreateKeyboardEvent,
    CGEventCreateScrollWheelEvent,
    CGEventPost,
    kCGEventLeftMouseDown,
    kCGEventLeftMouseUp,
    kCGEventLeftMouseDragged,
    kCGEventRightMouseDown,
    kCGEventRightMouseUp,
    kCGEventMouseMoved,
    kCGHIDEventTap,
    kCGMouseButtonLeft,
    kCGMouseButtonRight,
)


# Key name -> macOS keycode mapping (common keys)
_KEYCODES = {
    "return": 36,
    "enter": 36,
    "tab": 48,
    "space": 49,
    "delete": 51,
    "backspace": 51,
    "escape": 53,
    "esc": 53,
    "command": 55,
    "cmd": 55,
    "shift": 56,
    "option": 58,
    "alt": 58,
    "control": 59,
    "ctrl": 59,
    "left": 123,
    "right": 124,
    "down": 125,
    "up": 126,
    "a": 0, "b": 11, "c": 8, "d": 2, "e": 14, "f": 3, "g": 5,
    "h": 4, "i": 34, "j": 38, "k": 40, "l": 37, "m": 46, "n": 45,
    "o": 31, "p": 35, "q": 12, "r": 15, "s": 1, "t": 17, "u": 32,
    "v": 9, "w": 13, "x": 7, "y": 16, "z": 6,
    "0": 29, "1": 18, "2": 19, "3": 20, "4": 21, "5": 23,
    "6": 22, "7": 26, "8": 28, "9": 25,
}


def _get_keycode(key: str) -> int:
    """Get macOS keycode for a key name."""
    key_lower = key.lower()
    if key_lower in _KEYCODES:
        return _KEYCODES[key_lower]
    raise ValueError(f"Unknown key: {key}")


def _mouse_event(event_type: int, x: int, y: int, button: int = kCGMouseButtonLeft) -> None:
    """Create and post a mouse event."""
    point = Quartz.CGPointMake(x, y)
    event = CGEventCreateMouseEvent(None, event_type, point, button)
    CGEventPost(kCGHIDEventTap, event)


def click(x: int, y: int, button: str = "left") -> None:
    """Click at the given coordinates.

    Args:
        x: X coordinate.
        y: Y coordinate.
        button: Mouse button ("left", "right", "middle").
    """
    btn = kCGMouseButtonRight if button == "right" else kCGMouseButtonLeft
    event_down = kCGEventRightMouseDown if button == "right" else kCGEventLeftMouseDown
    event_up = kCGEventRightMouseUp if button == "right" else kCGEventLeftMouseUp

    _mouse_event(event_down, x, y, btn)
    time.sleep(0.05)
    _mouse_event(event_up, x, y, btn)


def double_click(x: int, y: int) -> None:
    """Double-click at the given coordinates."""
    click(x, y)
    time.sleep(0.05)
    click(x, y)


def move_mouse(x: int, y: int) -> None:
    """Move the mouse to the given coordinates."""
    _mouse_event(kCGEventMouseMoved, x, y)


def drag(start_x: int, start_y: int, end_x: int, end_y: int, duration: float = 0.5) -> None:
    """Drag from start to end coordinates.

    Args:
        start_x: Starting X coordinate.
        start_y: Starting Y coordinate.
        end_x: Ending X coordinate.
        end_y: Ending Y coordinate.
        duration: Duration of the drag in seconds.
    """
    # Move to start
    _mouse_event(kCGEventMouseMoved, start_x, start_y)
    time.sleep(0.05)

    # Mouse down
    _mouse_event(kCGEventLeftMouseDown, start_x, start_y)
    time.sleep(0.05)

    # Drag to end
    steps = max(10, int(duration / 0.02))
    for i in range(steps + 1):
        t = i / steps
        x = int(start_x + (end_x - start_x) * t)
        y = int(start_y + (end_y - start_y) * t)
        _mouse_event(kCGEventLeftMouseDragged, x, y)
        time.sleep(duration / steps)

    # Mouse up
    _mouse_event(kCGEventLeftMouseUp, end_x, end_y)


def scroll(x: int, y: int, scroll_x: int = 0, scroll_y: int = 0) -> None:
    """Scroll at the given coordinates.

    Args:
        x: X coordinate (for positioning).
        y: Y coordinate (for positioning).
        scroll_x: Horizontal scroll amount (positive = right).
        scroll_y: Vertical scroll amount (positive = down).
    """
    # Move mouse to position first
    _mouse_event(kCGEventMouseMoved, x, y)
    time.sleep(0.05)

    if scroll_y != 0:
        event = CGEventCreateScrollWheelEvent(None, 0, 1, scroll_y)
        CGEventPost(kCGHIDEventTap, event)

    if scroll_x != 0:
        event = CGEventCreateScrollWheelEvent(None, 0, 2, scroll_x)
        CGEventPost(kCGHIDEventTap, event)


def type_text(text: str, interval: float = 0.03) -> None:
    """Type the given text character by character.

    Args:
        text: Text to type.
        interval: Delay between keystrokes in seconds.
    """
    for char in text:
        keycode = _get_keycode(char.lower())
        # Determine if shift is needed for uppercase
        needs_shift = char.isupper() or char in '!@#$%^&*()_+{}|:"<>?'

        if needs_shift:
            shift_down = CGEventCreateKeyboardEvent(None, _KEYCODES["shift"], True)
            CGEventPost(kCGHIDEventTap, shift_down)

        key_down = CGEventCreateKeyboardEvent(None, keycode, True)
        key_up = CGEventCreateKeyboardEvent(None, keycode, False)
        CGEventPost(kCGHIDEventTap, key_down)
        CGEventPost(kCGHIDEventTap, key_up)

        if needs_shift:
            shift_up = CGEventCreateKeyboardEvent(None, _KEYCODES["shift"], False)
            CGEventPost(kCGHIDEventTap, shift_up)

        time.sleep(interval)


def keypress(keys: list[str]) -> None:
    """Press a combination of keys.

    Args:
        keys: List of key names to press simultaneously (e.g., ["cmd", "c"]).
    """
    keycodes = [_get_keycode(k) for k in keys]

    # Press all keys down
    for kc in keycodes:
        event = CGEventCreateKeyboardEvent(None, kc, True)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(0.02)

    # Release all keys in reverse order
    for kc in reversed(keycodes):
        event = CGEventCreateKeyboardEvent(None, kc, False)
        CGEventPost(kCGHIDEventTap, event)
        time.sleep(0.02)


def wait(seconds: float = 1.0) -> None:
    """Wait for the given number of seconds."""
    time.sleep(seconds)
