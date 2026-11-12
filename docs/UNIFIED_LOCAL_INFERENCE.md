# A.D.A. Unified Local Inference

## Target architecture

A.D.A. owns inference inside its existing Python/Qt process through
llama-cpp-python. It does not start Ollama, llama-server.exe, Docker, or a
second model-serving application.

Only one chat/vision model may be resident at a time:

1. K2-Horizon-7B-GGUF is the main coding and chat model.
2. When an image is attached, A.D.A. unloads K2 and loads Gemma 4.
3. Gemma receives the actual image bytes through the libmtmd multimodal path.
4. A.D.A. records Gemma's visual findings.
5. A.D.A. unloads Gemma and reloads K2.
6. K2 receives the user's original request plus Gemma's visual findings.

This does not make K2 a native vision model; A.D.A. orchestrates the vision
handoff at the application layer without a custom MCP server.

## Runtime compatibility

The pinned llama-cpp-python source commit vendors a current llama.cpp
submodule containing upstream K2-Horizon architecture support and Gemma 4
multimodal support. The Python API exposes Gemma4ChatHandler, which uses
libmtmd to process the image projector and image content.

Relevant upstream evidence:
- K2-Horizon support merged into llama.cpp PR #29535 on 2026-10-06.
- llama-cpp-python documents Gemma4ChatHandler and image_url inputs.
- A.D.A. uses the Python API in-process, not the optional web-server extra.

## Install the native runtime

From an x64 Native Tools PowerShell prompt for Visual Studio 2022, run:

    .\scripts\install_llama_cpp_python.ps1

For CPU-only builds:

    .\scripts\install_llama_cpp_python.ps1 -CpuOnly

CUDA builds require the CUDA Toolkit (nvcc) and MSVC. This script pins the
llama-cpp-python commit and initializes its llama.cpp submodule recursively.
It installs the Python bindings into the active environment, not a server.

## Model files

Configure local files in A.D.A. Settings:

- Chat model: a K2-Horizon-7B GGUF file.
- Vision model: a Gemma 4 12B GGUF file.
- Vision projector: Gemma 4's matching mmproj GGUF file.

The text model and projector must come from the same Gemma 4 GGUF release
family and be compatible with one another. Do not select a random mmproj file.

The default context length should remain conservative on a 12 GB RTX 5070.
Image tokens consume context and GPU memory.

## Memory behavior

EmbeddedModelRuntime serializes load, unload, and generation operations.
It explicitly closes the previous Llama object before creating the next.
Model weights and inference buffers still consume RAM/VRAM while a model is
loaded; embedding the runtime removes the extra server process, not the model's
memory requirement.

## Current implementation boundary

This is the initial runtime layer. Chat UI attachment plumbing, persisted
model-path settings, legacy Ollama removal, and lazy voice/router integration
must be migrated to this runtime before the migration is complete.
