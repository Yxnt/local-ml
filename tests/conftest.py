"""Root conftest — ensure server.agent is loaded before any test uses patch()."""

import os
import sys

# Add project root to sys.path before importing server.agent
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import importlib
importlib.import_module("server.agent")
