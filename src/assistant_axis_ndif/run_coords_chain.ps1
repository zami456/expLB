<#
  Extract evaluation-corpus activations unattended: prepare -> Pass B -> (Pass A, when asked).

  Runs on either machine. Repo root and venv are derived from $PSScriptRoot rather than
  hardcoded, because the lab PC keeps the tree at C:\Users\user4\expLB and the laptop at
  C:\Users\SURAIYA\Documents\STUDY\thesis\expLB, and the previous chain scripts hardcoded
  the laptop path -- copying one to the lab PC silently ran it against a path that does
  not exist there.

  Stage 0 is a no-NDIF corpus gate. --prepare_only builds and caches the assistant spans
  and prints per-dataset drop counts. It costs nothing and it catches the failure that
  matters most here: the lab PC's tree is a partial COPY, not a clone, so a missing CSV
  shows up as a dataset with zero rows rather than as an error. Spending an hour of NDIF
  time to discover that is the thing this stage exists to prevent.

  Pass B is the main corpus: 21 layers, all 15,358 rows, mean + last pooling, ~10 GB down.
  It is resumable at chunk granularity -- 8_coords_ndif.py counts the chunk_*.pt already on
  disk and skips the row_ids they cover -- so a kill, a sleep or an NDIF eviction costs only
  the session in flight.

  The layer list is every 4th layer and ALSO 22, not plain --layers every4. every4 yields
  0,4,...,76, which contains 20 and 40 but not 22 -- and 22 is Apollo's layer
  (roleplaying.yaml, descriptive.yaml), one of the three the layer question turns on,
  against Liars' Bench's 20 and the Assistant-Axis midpoint 40. Run as every4 the main
  corpus simply has no layer 22, so that three-way comparison could only ever be made on
  the 2,080-row Pass A subsample. The extra layer costs ~500 MB of the ~10 GB.

  Pass A (all 80 layers, subsampled) is deliberately NOT run here yet: --limit takes the
  FIRST N rows per dataset, which is not a stratified sample, and a label-sorted CSV would
  hand the layer sweep an imbalanced subsample. Fix the sampling before enabling it.

      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\run_coords_chain.ps1' -WindowStyle Hidden
#>
param(
    [string]$Out        = "run-coords",
    [int]$TokenBudget   = 12000,
    [int]$SessionBatches = 8,
    # every 4th layer, and also 22 (Apollo's) -- see the note above
    [int[]]$Layers      = @(0,4,8,12,16,20,22,24,28,32,36,40,44,48,52,56,60,64,68,72,76)
)
$ErrorActionPreference = "Continue"

$D    = $PSScriptRoot
$Repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $Repo

# .venv-aa on the lab PC, .venv-aa-win on the laptop; take whichever is actually here
$py = $null
foreach ($v in @(".venv-aa", ".venv-aa-win")) {
    $cand = Join-Path $Repo "$v\Scripts\python.exe"
    if (Test-Path $cand) { $py = $cand; break }
}
if (-not $py) { Write-Host "no windows venv found under $Repo"; exit 1 }

$log = Join-Path $Repo "logs\coords_chain.log"
New-Item -ItemType Directory -Force -Path (Join-Path $Repo "logs") | Out-Null
function Log($m) {
    $line = "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m"
    $line | Add-Content $log
    Write-Host $line
}

Log "=== coords chain started (pid $PID) repo=$Repo py=$py ==="

# ---- stage 0: corpus gate, no NDIF ---------------------------------------------------
$prep = Join-Path $Repo "logs\coords_prepare.log"
& $py "$D\8_coords_ndif.py" --datasets all --output_dir "$Out\passB" `
    --layers $Layers --prepare_only *> $prep
if ($LASTEXITCODE -ne 0) { Log "PREPARE FAILED (exit $LASTEXITCODE) -- see $prep"; exit 1 }
foreach ($line in (Get-Content $prep | Select-String 'rows|drop|span' | Select-Object -Last 20)) {
    Log "  prep| $line"
}

# ---- Pass B: every 4th layer plus 22, whole corpus -----------------------------------
Log "Pass B: $($Layers.Count) layers, all datasets, mean+last"
& $py "$D\8_coords_ndif.py" `
    --model meta-llama/Llama-3.3-70B-Instruct `
    --datasets all --output_dir "$Out\passB" `
    --layers $Layers --pooling mean last `
    --max_length 8192 --token_budget $TokenBudget --session_batches $SessionBatches `
    *>> (Join-Path $Repo "logs\coords_passB.log")
$code = $LASTEXITCODE
$chunks = (Get-ChildItem "$Out\passB\*\chunk_*.pt" -Recurse -EA SilentlyContinue).Count
Log "Pass B exited $code; $chunks chunk files on disk"
if ($code -eq 0) { New-Item -ItemType File -Force -Path "$Out\passB\.done" | Out-Null }

Log "=== coords chain finished ==="
