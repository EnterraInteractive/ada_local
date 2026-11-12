"""
In-process local model runtime for A.D.A.

Models are loaded inside the A.D.A. process through llama-cpp-python.
No Ollama service, llama-server.exe, or other inference daemon is started.

The manager enforces a single resident LLM: switching from K2 to Gemma closes
the active model before loading the next one.
"""

from __future__ import annotations

import base64
import gc
import mimetypes
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from core.settings_store import settings


class InferenceConfigurationError(RuntimeError):
    """Raised when local inference settings are incomplete or invalid."""


class InferenceUnavailableError(RuntimeError):
    """Raised when the embedded llama.cpp runtime is not installed."""


@dataclass(frozen=True)
class ModelProfile:
    key: str
    model_path: Path
    context_length: int
    gpu_layers: int
    projector_path: Optional[Path] = None


class EmbeddedModelRuntime:
    """
    Owns one llama.cpp model at a time, in the A.D.A. process.

    This is intentionally a process-local singleton. All model transitions
    and inference calls are serialized so two large models cannot be resident
    at once and the UI cannot race a model unload against active generation.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._llm: Any = None
        self._profile_key: Optional[str] = None
        self._loaded_profile: Optional[ModelProfile] = None
        self._chat_handler: Any = None
        self._last_error: Optional[str] = None

    @staticmethod
    def _profile(role: str) -> ModelProfile:
        if role == "chat":
            model_value = settings.get("models.chat_path", "")
            context = settings.get("models.chat_context", 8192)
            projector = None
        elif role == "vision":
            model_value = settings.get("models.vision_path", "")
            context = settings.get("models.vision_context", 8192)
            projector_value = settings.get("models.vision_projector_path", "")
            projector = Path(projector_value).expanduser() if projector_value else None
        else:
            raise ValueError(f"Unknown model role: {role}")

        if not model_value:
            raise InferenceConfigurationError(
                f"Select a local GGUF file for the {role} model in A.D.A. Settings."
            )

        model_path = Path(model_value).expanduser()
        if not model_path.is_file():
            raise InferenceConfigurationError(
                f"The configured {role} model file does not exist: {model_path}"
            )

        if role == "vision":
            if projector is None or not projector.is_file():
                raise InferenceConfigurationError(
                    "Select Gemma 4's mmproj GGUF file in A.D.A. Settings."
                )

        gpu_layers = int(settings.get("inference.gpu_layers", -1))
        return ModelProfile(
            key=role,
            model_path=model_path.resolve(),
            context_length=max(2048, int(context)),
            gpu_layers=gpu_layers,
            projector_path=projector.resolve() if projector else None,
        )

    @staticmethod
    def _llama_class():
        try:
            from llama_cpp import Llama
            return Llama
        except ImportError as exc:
            raise InferenceUnavailableError(
                "A.D.A.'s embedded llama.cpp runtime is not installed. "
                "Run scripts/install_llama_cpp_python.ps1 from the A.D.A. folder."
            ) from exc

    def _unload_locked(self) -> None:
        """Explicitly release the previous model before another is loaded."""
        old_llm = self._llm
        self._llm = None
        self._chat_handler = None
        self._profile_key = None
        self._loaded_profile = None

        if old_llm is not None:
            try:
                old_llm.close()
            finally:
                del old_llm
                gc.collect()

    def unload(self) -> None:
        with self._lock:
            self._unload_locked()

    def _load_locked(self, role: str):
        profile = self._profile(role)
        if self._llm is not None and self._loaded_profile == profile:
            return self._llm

        self._unload_locked()
        Llama = self._llama_class()

        cpu_threads = max(
            1,
            int(settings.get(
                "inference.cpu_threads",
                max(1, (os.cpu_count() or 4) - 2)
            ))
        )
        kwargs: Dict[str, Any] = {
            "model_path": str(profile.model_path),
            "n_ctx": profile.context_length,
            "n_gpu_layers": profile.gpu_layers,
            "n_threads": cpu_threads,
            "n_threads_batch": cpu_threads,
            "n_batch": 512,
            "n_ubatch": 128,
            "use_mmap": True,
            "use_mlock": False,
            "verbose": False,
        }

        handler = None
        if role == "vision":
            try:
                from llama_cpp.llama_chat_format import Gemma4ChatHandler
            except ImportError as exc:
                raise InferenceUnavailableError(
                    "The installed embedded llama.cpp build does not expose "
                    "Gemma4ChatHandler. Rebuild A.D.A.'s pinned runtime."
                ) from exc

            handler = Gemma4ChatHandler(
                clip_model_path=str(profile.projector_path),
                verbose=False,
                use_gpu=True,
            )
            kwargs["chat_handler"] = handler

        try:
            llm = Llama(**kwargs)
        except Exception:
            if handler is not None:
                del handler
            gc.collect()
            raise

        self._llm = llm
        self._chat_handler = handler
        self._profile_key = role
        self._loaded_profile = profile
        self._last_error = None
        return llm

    def stream_chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        max_tokens: int = 2048,
        temperature: float = 0.4,
        top_p: float = 0.9,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[str]:
        """Stream K2 chat text while keeping the single-model lock."""
        with self._lock:
            llm = self._load_locked("chat")
            stream = llm.create_chat_completion(
                messages=messages,
                stream=True,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            for chunk in stream:
                if stop_event is not None and stop_event.is_set():
                    break
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if isinstance(content, str) and content:
                    yield content

    @staticmethod
    def _image_data_uri(image_path: str) -> str:
        path = Path(image_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Image attachment not found: {path}")
        if path.stat().st_size > 25 * 1024 * 1024:
            raise ValueError(f"Image is larger than A.D.A.'s 25 MB limit: {path.name}")

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type or not mime_type.startswith("image/"):
            raise ValueError(f"Unsupported image attachment: {path.name}")

        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @staticmethod
    def _base64_image_data_uri(image_b64: str, mime_type: str = "image/jpeg") -> str:
        """Convert a raw screenshot base64 payload to a multimodal data URI."""
        value = image_b64.strip()
        if value.startswith("data:"):
            return value
        if not value:
            raise ValueError("The supplied screenshot is empty.")
        try:
            raw = base64.b64decode(value, validate=True)
        except Exception as exc:
            raise ValueError("The supplied screenshot is not valid base64.") from exc
        if len(raw) > 25 * 1024 * 1024:
            raise ValueError("Screenshot exceeds A.D.A.'s 25 MB image limit.")
        if not mime_type.startswith("image/"):
            raise ValueError("Image MIME type must start with image/.")
        return f"data:{mime_type};base64,{value}"

    def vision_completion(
        self,
        messages: List[Dict[str, Any]],
        *,
        image_paths: Iterable[str] = (),
        image_base64: Iterable[str] = (),
        max_tokens: int = 1600,
        temperature: float = 0.2,
        top_p: float = 0.9,
        keep_vision_loaded: bool = False,
        restore_chat_model: bool = True,
    ) -> str:
        """
        Run a multimodal Gemma completion inside A.D.A.

        keep_vision_loaded=True is for multi-step browser-agent sessions:
        Gemma stays loaded between screenshot/action turns and is released when
        the task ends. Normal chat attachment analysis unloads Gemma immediately.
        """
        paths = list(image_paths)
        encoded_images = list(image_base64)
        if not paths and not encoded_images:
            raise ValueError("Vision analysis requires at least one image.")

        with self._lock:
            previous_role = self._profile_key
            llm = self._load_locked("vision")
            try:
                prepared = [dict(message) for message in messages]
                last_user_index = next(
                    (i for i in range(len(prepared) - 1, -1, -1)
                     if prepared[i].get("role") == "user"),
                    None,
                )
                if last_user_index is None:
                    prepared.append({"role": "user", "content": ""})
                    last_user_index = len(prepared) - 1

                original_content = prepared[last_user_index].get("content", "")
                text_content = (
                    original_content if isinstance(original_content, str)
                    else str(original_content)
                )
                multimodal_content: List[Dict[str, Any]] = [
                    {"type": "text", "text": text_content}
                ]

                for image_path in paths:
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {"url": self._image_data_uri(image_path)},
                    })
                for image in encoded_images:
                    multimodal_content.append({
                        "type": "image_url",
                        "image_url": {"url": self._base64_image_data_uri(image)},
                    })

                prepared[last_user_index]["content"] = multimodal_content
                result = llm.create_chat_completion(
                    messages=prepared,
                    stream=False,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                message = result["choices"][0]["message"]
                answer = message.get("content", "")
                if not isinstance(answer, str) or not answer.strip():
                    raise RuntimeError("Gemma returned an empty visual analysis.")
                return answer.strip()
            finally:
                if not keep_vision_loaded:
                    self._unload_locked()
                    if restore_chat_model and previous_role == "chat":
                        self._load_locked("chat")

    def analyze_images(
        self,
        image_paths: Iterable[str],
        instruction: str,
        *,
        stop_event: Optional[threading.Event] = None,
        restore_chat_model: bool = True,
    ) -> str:
        """Analyze chat attachments, then release Gemma before K2 responds."""
        if stop_event is not None and stop_event.is_set():
            return ""
        return self.vision_completion(
            [
                {
                    "role": "system",
                    "content": (
                        "You are A.D.A.'s visual analysis model. Describe only "
                        "evidence present in the supplied images. For Unreal Editor "
                        "screenshots, identify panels, settings, warnings, errors, "
                        "and visible object state. Separate observations from inferences."
                    ),
                },
                {
                    "role": "user",
                    "content": instruction or "Describe the attached image(s).",
                },
            ],
            image_paths=image_paths,
            max_tokens=1600,
            temperature=0.2,
            top_p=0.9,
            keep_vision_loaded=False,
            restore_chat_model=restore_chat_model,
        )

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "runtime": "llama.cpp (embedded)",
                "loaded_role": self._profile_key,
                "model_loaded": self._llm is not None,
                "last_error": self._last_error,
            }


runtime = EmbeddedModelRuntime()
