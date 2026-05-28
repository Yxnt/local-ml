"""DSPy-based automatic optimization for the local AI agent.

Collects real usage data and uses DSPy to automatically improve:
- System prompts
- Tool descriptions
- Response strategies
"""

from optimization.collector import UsageCollector, Interaction
from optimization.optimizer import PromptOptimizer
from optimization.trainer import AutoTrainer

__all__ = ["UsageCollector", "Interaction", "PromptOptimizer", "AutoTrainer"]
