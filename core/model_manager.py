"""
Embedded model lifecycle compatibility helpers.

There is no external model server. All operations target the one in-process
llama.cpp model instance owned by A.D.A.
"""

from core.inference import runtime


def sync_unload_model(model_name: str = "") -> None:
    """Release whichever embedded model is currently resident."""
    runtime.unload()


def unload_model(model_name: str = "") -> None:
    """Release the embedded model synchronously."""
    runtime.unload()


def unload_all_models(sync: bool = False) -> None:
    """Release A.D.A.'s single embedded model."""
    runtime.unload()


def get_running_models() -> list:
    """Return the currently loaded model role, if any."""
    status = runtime.status()
    role = status.get("loaded_role")
    return [role] if role else []


def ensure_exclusive_qwen(target_model: str = "") -> None:
    """Compatibility shim; embedded runtime already enforces one model."""
    return None
