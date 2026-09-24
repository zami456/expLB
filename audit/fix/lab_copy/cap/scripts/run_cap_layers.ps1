<#
  The cap result across ALL 21 stored layers, not the 3 it was measured at.

  WHY. F-72 measured F-03 -- the project's strongest positive result, quoted at a single layer --
  across depth and found it collapses from 0.739 at L40 to 0.374 at L20, a span of 0.365 against
  a claimed margin over raw of +0.033. The layer choice moved that headline by eleven times the
  size of the effect it reported.

  F-66 and F-68 carry the same exposure. The cap bank's advantage (+0.112 over raw on
  sycophancy_mt, CI [+0.050, +0.185]) was measured at layers 20, 40 and 44 -- three of the
  twenty-one that passB stores. L20 already failed to clear against datapca and against
  cap_only, so the depth dependence is not hypothetical; it is visible in the three points we
  have. Eighteen more are sitting on disk.

  COSTS NO NDIF. The cap axes are all 80 layers (run-cap/axes_s23.pt) and the evaluation
  activations are passB's 21. Both exist. This is CPU only.

  WHAT WOULD CHANGE THE CLAIM. If the margin over raw holds across most of the 21 layers, F-66
  is a depth-robust result and the reviewers' "contested novel" verdict on it gets stronger. If
  it is confined to a narrow band around L40, it is the same shape as F-03 and must be reported
  that way -- which would also mean the format-matched persona control (the next proposed
  experiment) would be defending a result that is itself fragile.

  Prediction written down before the run, per the standing rule: the margin will be POSITIVE
  across roughly L22-L48 and will fall off at both ends, with the late layers (L60+) closer to
  zero than negative. If instead it is positive at only 3-5 layers, F-66 needs the same
  treatment F-72 gave F-03.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "cap_layers.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+DONE\s*$' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")
$FEATS = @("raw", "random", "length", "datapca", "random@2", "datapca@2",
           "pc:capitulation", "pc:cap_vs_default", "pc:cap_only",
           "pc:capitulation@2", "pc:cap_vs_default@2", "pc:cap_only@2")

# Every layer passB stores. 20, 40 and 44 already exist in the cap_s23/cap_all banks, so those
# are skipped by the resume logic and only the other eighteen are computed.
foreach ($v in @("s23", "all")) {
    $bank = Join-Path $repo "run-cap\axes_$v.pt"
    if (-not (Test-Path $bank)) { Note "MISSING $bank -- skipping $v"; continue }
    $t0 = Get-Date
    Note "cap_$v : all stored layers"
    & $pyAN "$SRC\27_crossgen.py" --run_dir "run-coords\passB" `
        --out_root "run-crossgen\kd11\banks\cap_$v" --bank_name "cap_$v" --bank $bank `
        --baseline_root "run-crossgen\kd11" --datasets @DS --features @FEATS `
        --stage all *>> (Join-Path $logDir "cap_layers_$v.log")

    # Gate on the number of DISTINCT LAYERS present, not the file count: the run is pointless
    # unless it actually widened the layer coverage from the three we had.
    $mdir = Join-Path $repo "run-crossgen\kd11\banks\cap_$v\matrices"
    $files = @(Get-ChildItem $mdir -Filter "mean_L*_s42.json" -EA SilentlyContinue)
    $layers = @($files | ForEach-Object {
        if ($_.Name -match 'mean_L(\d+)_s42\.json') { [int]$Matches[1] } }) | Sort-Object -Unique
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    Note ("cap_{0}  exit {1}  mean-pooled layers={2} (was 3, predicted 21)  {3} min  {4}" -f
          $v, $LASTEXITCODE, $layers.Count, $mins,
          $(if ($layers.Count -ge 21) { "PASS" } else { "GATE FAILED" }))
    Note ("cap_{0}  layers: {1}" -f $v, ($layers -join ','))
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "cap_layers/final" -EA SilentlyContinue
Note "DONE"
