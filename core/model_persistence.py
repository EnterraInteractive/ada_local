"""
Compatibility wrappers for the former Ollama model persistence manager.

A.D.A. has one embedded inference runtime and does not keep a timer thread
polling an external model server.
"""

from core.inference import runtime


def ensure_qwen_loaded() -> bool:
    """Validate that the primary local model is configured."""
    try:
        runtime._profile("chat")
        return True
    except Exception:
        return False


def mark_qwen_used() -> None:
    """No-op: the embedded model manager owns model lifetime."""
    return None


def unload_qwen(reason: str = "manual") -> None:
    """Release the currently loaded embedded model."""
    runtime.unload()


def get_qwen_status() -> dict:
    """Return the embedded model status using the legacy result shape."""
    status = runtime.status()
    return {
        "is_loaded": status.get("model_loaded", False),
        "is_running": status.get("model_loaded", False),
        "last_used": None,
        "time_since_use": None,
        "loaded_role": status.get("loaded_role"),
        "runtime": status.get("runtime"),
    }
