"""
Compatibility helpers for the legacy intent-router interface.

A.D.A. now uses its primary embedded K2 model and a small deterministic
command parser. This module deliberately does not load FunctionGemma, PyTorch,
Ollama, or a second model.
"""

from config import GRAY, RESET


def is_router_loaded() -> bool:
    """The separate FunctionGemma router is no longer loaded by A.D.A."""
    return False


def should_bypass_router(text: str) -> bool:
    """Always use the main embedded assistant; no separate router is needed."""
    return True


def route_query(user_input: str):
    """Legacy-compatible fallback that routes normal chat to K2."""
    return "nonthinking", {"prompt": user_input}


def execute_function(name: str, params: dict) -> str:
    """Legacy display helper; actual actions use core.function_executor."""
    return f"Requested action: {name} ({params})"


def preload_models() -> None:
    """No models are preloaded; A.D.A. loads a single model on demand."""
    print(f"{GRAY}[System] Local models load on demand.{RESET}")
