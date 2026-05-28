"""Screen capture using macOS Quartz/CoreGraphics."""

import base64
import tempfile
from pathlib import Path

import Quartz
from Quartz import (
    CGWindowListCreateImage,
    CGMainDisplayID,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    kCGImageAlphaPremultipliedLast,
    CGImageDestinationCreateWithURL,
    CGImageDestinationAddImage,
    CGImageDestinationFinalize,
)
import Foundation


def capture_screen(output_path: str | Path | None = None) -> bytes:
    """Capture the full screen and return PNG bytes.

    Args:
        output_path: Optional path to save the screenshot. If None, uses a temp file.

    Returns:
        PNG image bytes.
    """
    # Get the main display bounds
    main_display = CGMainDisplayID()
    bounds = Quartz.CGDisplayBounds(main_display)

    # Capture the screen
    image = CGWindowListCreateImage(
        bounds,
        kCGWindowListOptionOnScreenOnly,
        kCGNullWindowID,
        kCGImageAlphaPremultipliedLast,
    )

    if image is None:
        raise RuntimeError(
            "Failed to capture screen. Grant Screen Recording permission to Terminal/Python."
        )

    # Determine output path
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        output_path = tmp.name
        tmp.close()

    output_path = str(output_path)

    # Convert to PNG via CGImageDestination
    url = Foundation.CFURLCreateWithFileSystemPath(
        None,
        output_path,
        Foundation.kCFURLPOSIXPathStyle,
        False,
    )
    dest = CGImageDestinationCreateWithURL(url, "public.png", 1, None)
    if dest is None:
        raise RuntimeError("Failed to create image destination")

    CGImageDestinationAddImage(dest, image, None)
    CGImageDestinationFinalize(dest)

    return Path(output_path).read_bytes()


def capture_screen_base64() -> str:
    """Capture the screen and return a base64-encoded PNG string.

    Returns:
        Base64-encoded PNG image string.
    """
    png_bytes = capture_screen()
    return base64.b64encode(png_bytes).decode("utf-8")
