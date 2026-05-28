"""Simple CLI WebSocket client for testing.

Usage:
    python server/ws_client.py [--url ws://localhost:8000/ws] [--session SESSION_ID]

Requirements:
    pip install websockets
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

try:
    import websockets
except ImportError:
    print("Install websockets: pip install websockets")
    sys.exit(1)


async def run_client(url: str, session_id: str | None = None) -> None:
    """Run interactive WebSocket client.

    Args:
        url: WebSocket server URL.
        session_id: Optional session ID for conversation tracking.
    """
    print(f"Connecting to {url}...")
    print("Type your message and press Enter. Type 'quit' or Ctrl+C to exit.")
    print("Commands: /model <name> - switch model, /clear - clear session")
    print("-" * 60)

    async with websockets.connect(url) as ws:
        current_model = None

        while True:
            try:
                # Get user input
                message = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: input("> ")
                )
            except (EOFError, KeyboardInterrupt):
                print("\nDisconnecting...")
                break

            message = message.strip()
            if not message:
                continue

            if message.lower() in ("quit", "exit", "q"):
                print("Disconnecting...")
                break

            # Handle commands
            if message.startswith("/model "):
                model_name = message[7:].strip()
                if model_name:
                    current_model = model_name
                    print(f"Model set to: {current_model}")
                else:
                    print(f"Current model: {current_model or 'default'}")
                continue

            if message == "/clear":
                # Generate new session to effectively clear history
                session_id = None
                print("Session cleared (new session will be created)")
                continue

            # Build request
            request = {
                "type": "text",
                "content": message,
            }
            if current_model:
                request["model"] = current_model
            if session_id:
                request["session_id"] = session_id

            # Send message
            await ws.send(json.dumps(request))

            # Receive responses (server sends processing status then result)
            while True:
                response_raw = await ws.recv()
                response = json.loads(response_raw)

                status = response.get("status")

                if status == "processing":
                    # Wait for actual response
                    continue

                # Print response
                if status == "error":
                    print(f"[Error] {response.get('content', 'Unknown error')}")
                else:
                    content = response.get("content", "")
                    if content:
                        print(content)

                    # Show tool calls if present
                    tool_calls = response.get("tool_calls")
                    if tool_calls:
                        print(f"[Tool calls: {len(tool_calls)}]")
                        for tc in tool_calls:
                            fn = tc.get("function", {})
                            print(f"  - {fn.get('name')}({fn.get('arguments', '{}')})")

                # Update session_id from response
                if "session_id" in response:
                    session_id = response["session_id"]

                break

        print("Disconnected.")


def main() -> None:
    parser = argparse.ArgumentParser(description="WebSocket client for local-ml")
    parser.add_argument(
        "--url",
        default="ws://localhost:8000/ws",
        help="WebSocket server URL (default: ws://localhost:8000/ws)",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Session ID for conversation tracking",
    )
    args = parser.parse_args()

    asyncio.run(run_client(args.url, args.session))


if __name__ == "__main__":
    main()
