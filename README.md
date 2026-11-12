# A.D.A. — Advanced Digital Assistant

A.D.A. is a local Windows desktop assistant built with Python and PySide6.
The application owns its model runtime; it does not require Ollama, LM Studio,
Open WebUI, Jan, or another model-serving application.

## Unified local inference

A.D.A. uses an embedded llama.cpp runtime through llama-cpp-python. The
inference library runs inside the A.D.A. application process. It does not
launch a separate llama-server.exe or a local model-server service.

A.D.A. is designed to keep only one large language model resident at a time:

- **K2-Horizon 7B Unreal Engine GGUF** — primary chat, coding, and reasoning.
- **Gemma 4 12B GGUF** — loaded on demand for screenshot and image analysis.
- **Gemma 4 mmproj GGUF** — the matching multimodal projector for that exact
  Gemma model release.

When an image is attached in chat, A.D.A. loads Gemma, passes it the actual
image, collects its visual findings, unloads it, and then asks K2 to answer
using those findings. K2 itself remains a text model; the vision handoff is
handled internally by A.D.A.

The browser agent uses the same embedded Gemma model and releases it when its
browser task ends. Voice input uses the same K2 model. The application does
not preload a separate FunctionGemma intent model.

> Embedded inference removes a separate model-server process, not the memory
> required by the model itself. Model weights, context cache, and GPU buffers
> still use system RAM and VRAM while inference is active.

## Current application features

- Local chat with streaming responses and persistent conversation history.
- Image and screenshot attachments in chat, analyzed by Gemma 4.
- Browser automation using the embedded Gemma vision model.
- Optional wake-word speech input and text-to-speech.
- Timers, alarms, calendar, tasks, web search, weather, and news briefing.
- TP-Link Kasa smart-device controls.
- CPU, memory, and GPU monitoring.
- Single-model loading and explicit model release.

## Windows setup

### 1. Install prerequisites

Install:

- Python 3.11 (64-bit)
- Git for Windows
- CMake
- Visual Studio 2022 Build Tools with **Desktop development with C++**
- NVIDIA CUDA Toolkit compatible with your installed NVIDIA driver (for CUDA inference)

Run the following commands from an **x64 Native Tools PowerShell prompt for
Visual Studio 2022** so MSVC and the Windows SDK are available.

### 2. Create the Python environment

    py -3.11 -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt

### 3. Build the embedded inference runtime

For NVIDIA CUDA acceleration:

    .\scripts\install_llama_cpp_python.ps1

For a CPU-only build:

    .\scripts\install_llama_cpp_python.ps1 -CpuOnly

The script checks for Git, CMake, MSVC, and (for CUDA builds) the CUDA
compiler. It checks out a pinned llama-cpp-python source revision with its
llama.cpp submodule. It installs Python bindings into the active A.D.A.
environment; it does not install or run a separate inference server.

### 4. Install the browser engine

    python -m playwright install chromium

### 5. Configure the models

Start A.D.A.:

    python main.py

Open **Settings → Local AI Models** and select:

1. Your K2-Horizon 7B Unreal Engine GGUF file as the Primary Chat Model.
2. Your Gemma 4 12B GGUF file as the Vision Model.
3. The matching Gemma 4 mmproj GGUF file as the Vision Projector.

The Gemma text model and projector must be compatible files from the same
model release. Do not select an unrelated projector.

Start with conservative context lengths (the default is 8192) and adjust GPU
layers if the combined model, projector, context, and Unreal Editor exceed
available VRAM.

### 6. Optional voice input

Voice input is disabled by default so speech-recognition models are not loaded
while you are developing.

To install voice dependencies:

    pip install -r requirements-voice.txt

Then enable **Settings → Voice & Audio → Voice Assistant**. Speech recognition
loads only after you turn it on. Voice responses use the configured K2 model.

## Model and memory settings

- **K2 Context Length** controls the primary model's context allocation.
- **Gemma Context Length** controls the vision model's context allocation.
- **GPU Layers** controls model offload. Use \`-1\` for all layers where memory
  permits, or lower the number to keep more layers in system RAM.
- **CPU Threads** controls inference CPU thread count.

A.D.A. serializes model loading and inference. When switching from K2 to
Gemma, A.D.A. closes the previous model before loading the next one.

## Development notes

The legacy FunctionGemma router and Ollama adapter files are no longer part
of the desktop inference path. Optional development dependencies for the old
router and training scripts are isolated in \`requirements-router-dev.txt\`.

The embedded runtime and image handoff are on the
\`feature/ada-unified-local-ai\` branch. Main has not been changed.

## Troubleshooting

### Embedded runtime is unavailable

Run the installer from the active virtual environment and check the CMake
output. CUDA builds require the CUDA Toolkit compiler (nvcc) and a working
MSVC developer environment.

### Gemma reports that vision is not supported

Confirm that the selected Gemma GGUF and mmproj GGUF are a matching pair and
that the installed llama.cpp binding was built from A.D.A.'s pinned revision.

### CUDA out of memory

Reduce the GPU layer count or context length. Close other applications using
the GPU. A.D.A. loads K2 and Gemma sequentially, but it cannot eliminate their
individual memory requirements.

### Voice assistant fails to initialize

Confirm that requirements-voice.txt is installed, microphone permissions
are enabled, and the configured speech-recognition model can be downloaded.
