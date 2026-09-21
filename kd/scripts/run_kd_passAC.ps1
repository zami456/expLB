<#
  The 10-dataset grid on the passA and passC stores, root then kd bank.

  WHY. Only passB has been scored with 11 datasets so far. passA (all 80 layers, 2,000
  stratified rows) gives the layer profile; passC (per-token, 6 layers, 2,000 rows) is the only
  store that can answer whether an axis fires on particular TOKENS within a response.

  NO GPU AND NO NDIF. 27_crossgen.py reads stored activations and trains probes; it contains no
  nnsight and runs in .venv-analysis. The persona bank supplies DIRECTIONS, and applying a
  direction to an already-extracted activation is a dot product. Nothing is re-extracted.

  TEN datasets, not eleven. passA and passC are 2,000-row subsamples and their HP-KR rows are
  label-stratified 100/100, which leaves the matched slice at 89 deceptive / 9 honest -- see
  the comment above $DS. The matched-slice column is reportable on passB ONLY.

  Each store needs its own root (baselines: raw, random, dataset PCA, length) before a bank can
  be merged against it -- crossgen_root() resolves <root>/banks/<name> back to <root>, so the
  root must exist first. Hence four steps, in order.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRCDIR = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "kd_passAC.status"
$log = Join-Path $logDir "kd_passAC.log"
$bank = Join-Path $repo "run-kd\axes_vectors_s23.pt"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if (-not $Force -and (Test-Path $status) -and
    ((Get-Content $status -EA SilentlyContinue) -match '^DONE ').Count -gt 0) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status
if (-not (Test-Path $bank)) { Note "MISSING $bank -- stopping"; exit 1 }

# TEN datasets here, NOT eleven. harm-pressure-kr-matched is deliberately excluded from passA
# and passC, and this is a correctness matter rather than a preference.
#
# Measured 2026-09-20: passA and passC label-stratify harm-pressure-knowledge-report down to
# 100 deceptive / 100 honest. But HP-KR's honest class is overwhelmingly "I know the answer" --
# assertions, which by construction cannot appear in a DENIAL-matched slice. So the matched
# view there is 98 rows at 89 deceptive / 9 honest, against passB's balanced 124 / 134.
#
# An AUROC computed against nine negatives is noise with a number attached, and it would have
# been printed in a heatmap cell indistinguishable from a real one. passB is the only store
# where the matched slice is reportable.
$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# layers: passA stores all 80, passC only its six. Start at the layers every conclusion is read
# at, so partial results are useful if the run is interrupted.
# Outputs nest UNDER run-crossgen\kd11 on purpose: sync_kd.ps1 already pushes that whole
# tree to Drive, and crossgen_root() resolves <root>/banks/<name> back to <root>, so
# run-crossgen\kd11\passA and its banks/ subfolder both resolve correctly.
# Peak layers FIRST so partial results are usable, then the FULL sweep. passA stores all 80
# layers and the layer profile is its entire purpose -- restricting it to three would discard
# exactly what it is for. passC has only its six, so one full pass covers it.
# An empty layers list means "every stored layer" (27_crossgen.py --layers default=[]).
$steps = @(
    @{ tag = "A-root";     run = "run-coords\passA"; out = "run-crossgen\kd11\passA"; layers = @("40","44","20"); bank = $false },
    @{ tag = "A-kd";       run = "run-coords\passA"; out = "run-crossgen\kd11\passA\banks\kd_s23"; layers = @("40","44","20"); bank = $true },
    @{ tag = "C-root";     run = "run-coords\passC"; out = "run-crossgen\kd11\passC"; layers = @(); bank = $false },
    @{ tag = "C-kd";       run = "run-coords\passC"; out = "run-crossgen\kd11\passC\banks\kd_s23"; layers = @(); bank = $true },
    @{ tag = "A-root-all"; run = "run-coords\passA"; out = "run-crossgen\kd11\passA"; layers = @(); bank = $false },
    @{ tag = "A-kd-all";   run = "run-coords\passA"; out = "run-crossgen\kd11\passA\banks\kd_s23"; layers = @(); bank = $true }
)

foreach ($step in $steps) {
    Note "STEP $($step.tag) layers=$($step.layers -join ',') start"
    $a = @("$SRCDIR\27_crossgen.py", "--run_dir", $step.run, "--out_root", $step.out,
           "--datasets") + $DS
    # Omit --layers entirely when the list is empty. A bare switch with no value would swallow
    # the NEXT argument -- the same empty-argument hazard that silently killed two crossgen
    # steps on 2026-09-19 and made the chain report DONE having run half its work.
    if ($step.layers.Count) { $a += @("--layers") + $step.layers }
    if ($step.bank) {
        $a += @("--bank_name", "kd_s23", "--bank", $bank,
                "--baseline_root", (Split-Path (Split-Path $step.out -Parent) -Parent),
                "--features", "pc:deception", "axes", "axes_resid", "pc:unified",
                              "pc:knowledge_denial", "pc:kd_assertion",
                "--extra_features", "axes", "pc:knowledge_denial", "pc:kd_assertion")
    }
    $a += @("--stage", "all")
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    Note "STEP $($step.tag) exit $LASTEXITCODE"
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Wait -Tag "passAC/final"
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
