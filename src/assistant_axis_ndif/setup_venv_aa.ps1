# Create the isolated Python env for the NDIF Assistant-Axis pipeline (Windows).
#
#   powershell -ExecutionPolicy Bypass -File src\assistant_axis_ndif\setup_venv_aa.ps1
#
# Makes  <repo>\.venv-aa  (CPU-only, ~1.1 GB). Does NOT touch the repo's .venv / uv.lock.
# Requires `uv`  (winget install astral-sh.uv, or https://astral.sh/uv).
#
# Mirrors setup_venv_aa.sh exactly -- same pinned versions. The socket pins are
# load-bearing: websocket-client 1.9.x + python-engineio 4.14 crash the NDIF result
# socket on long jobs ("invalid literal for int() ... b'P'").
param(
    [string]$VenvPath = ""
)
$ErrorActionPreference = "Stop"

$repo = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
if (-not $VenvPath) { $VenvPath = Join-Path $repo ".venv-aa" }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "install uv first:  winget install astral-sh.uv   (or see https://astral.sh/uv)"
    exit 1
}

Write-Host "=== creating $VenvPath (python 3.12) ==="
uv venv --python 3.12 $VenvPath

$py = Join-Path $VenvPath "Scripts\python.exe"

Write-Host "=== installing pinned deps (CPU torch) ==="
$env:UV_LINK_MODE = "copy"
uv pip install --python $py `
    --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match `
    "nnsight==0.7.0" `
    "torch==2.14.0" `
    "transformers==5.17.0" `
    "websocket-client==1.8.0" `
    "huggingface_hub" `
    "python-dotenv" `
    "jsonlines" `
    "tqdm"

# The verified socket combo is python-socketio 5.16.4 + python-engineio 4.13.0, but
# socketio 5.16.4's metadata declares engineio>=4.13.2, so a strict resolver (uv)
# refuses it. pip installed it anyway in the devcontainer and it works. Force the exact
# pins with --no-deps so the metadata floor is bypassed.
Write-Host "=== pinning the verified socket combo (--no-deps) ==="
uv pip install --python $py --no-deps --reinstall `
    "python-socketio==5.16.4" `
    "python-engineio==4.13.0"

Write-Host ""
Write-Host "=== done. quick import check (first run is slow, ~3 min cold) ==="
& $py -c "import time; t=time.time(); import torch, transformers, nnsight; print(f'torch {torch.__version__} | transformers {transformers.__version__} | nnsight {nnsight.__version__}  ({time.time()-t:.0f}s)')"

Write-Host ""
Write-Host "Next: copy src\assistant_axis_ndif\env.aa.example src\assistant_axis_ndif\.env  and fill it in."
