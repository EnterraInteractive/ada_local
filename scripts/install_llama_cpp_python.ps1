# Build and install A.D.A.'s embedded llama.cpp runtime.
# This installs Python bindings into the active A.D.A. environment.
# It does NOT install Ollama, Docker, or llama-server.exe.
#
# Requirements:
# - Git
# - CMake
# - Visual Studio 2022 C++ build tools (run from an x64 Native Tools prompt)
# - CUDA Toolkit compatible with the installed NVIDIA driver (for CUDA build)
#
# The pinned llama-cpp-python commit includes:
# - llama.cpp K2-Horizon support
# - libmtmd / Gemma 4 image support

param(
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"

$PinnedCommit = "05c124e33f7f33b7e66ef887274ad9ee6775099f"
$BuildRoot = Join-Path $PSScriptRoot "..\.build"
$SourceDir = Join-Path $BuildRoot "llama-cpp-python"

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "Git is required. Install Git for Windows and rerun this script."
}

if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    throw "CMake is required. Install CMake and rerun this script."
}

if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
    throw "Run this script from 'x64 Native Tools Command Prompt for VS 2022' (or an equivalent MSVC developer shell)."
}

if (-not $CpuOnly -and -not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "CUDA Toolkit compiler (nvcc) was not found. Install a CUDA Toolkit or rerun with -CpuOnly."
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

if (-not (Test-Path (Join-Path $SourceDir ".git"))) {
    git clone --recurse-submodules https://github.com/abetlen/llama-cpp-python.git $SourceDir
    if ($LASTEXITCODE -ne 0) { throw "Failed to clone llama-cpp-python." }
}

git -C $SourceDir fetch origin main
if ($LASTEXITCODE -ne 0) { throw "Failed to fetch llama-cpp-python." }

git -C $SourceDir checkout $PinnedCommit
if ($LASTEXITCODE -ne 0) { throw "Failed to checkout pinned llama-cpp-python commit." }

git -C $SourceDir submodule sync --recursive
if ($LASTEXITCODE -ne 0) { throw "Failed to sync llama.cpp submodule." }

git -C $SourceDir submodule update --init --recursive
if ($LASTEXITCODE -ne 0) { throw "Failed to update llama.cpp submodule." }

python -m pip install --upgrade pip scikit-build-core cmake ninja
if ($LASTEXITCODE -ne 0) { throw "Failed to install native build tools." }

if ($CpuOnly) {
    $env:CMAKE_ARGS = "-DGGML_CUDA=OFF"
} else {
    $env:CMAKE_ARGS = "-DGGML_CUDA=ON"
}
$env:FORCE_CMAKE = "1"

python -m pip install --upgrade --force-reinstall --no-build-isolation $SourceDir
if ($LASTEXITCODE -ne 0) {
    throw "llama-cpp-python build failed. Review the CMake output above."
}

python -c "from llama_cpp import Llama; from llama_cpp.llama_chat_format import Gemma4ChatHandler; print('A.D.A embedded llama.cpp runtime: READY')"
if ($LASTEXITCODE -ne 0) {
    throw "The installed binding did not expose the required K2/Gemma 4 runtime API."
}

Write-Host ""
Write-Host "A.D.A embedded runtime installed. No separate inference server was installed."
