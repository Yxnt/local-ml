"""Demo: Computer use with local models.

Usage:
    # With default model (gemma-4-e2b-it-4bit)
    python computer_use_demo.py "Open Safari and go to example.com"

    # With specific model
    python computer_use_demo.py --model minicpm5-1b-mlx "Take a screenshot"

    # With MiniCPM-V 4.6 (recommended for computer use)
    python computer_use_demo.py --model minicpm-v-4_6 "Take a screenshot and describe what you see"
"""

import argparse
import asyncio

from model.backends import ModelRegistry
from computer_use.agent import ComputerUseAgent


async def main():
    parser = argparse.ArgumentParser(description="Computer use demo with local models")
    parser.add_argument("task", help="Task description for the agent")
    parser.add_argument("--model", default="gemma-4-e2b-it-4bit", help="Model to use")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum steps")
    parser.add_argument("--screenshots", default="./screenshots", help="Directory to save screenshots")
    args = parser.parse_args()

    # Initialize registry and load model
    registry = ModelRegistry()
    registry.register_defaults()

    print(f"Loading model: {args.model}")
    backend = await registry.get_or_load(args.model)
    print("Model loaded!")

    # Create agent and run
    agent = ComputerUseAgent(backend, max_steps=args.max_steps)
    result = agent.run(args.task, screenshot_dir=args.screenshots)

    print("\n=== Result ===")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
