<#
  The pooled-training result across every layer passB stores.

  WHY. F-79/F-80 measured pooled leave-one-dataset-out transfer at 0.8545 against single-source
  0.6763, +0.178 with a 95% dataset-level CI of [+0.143, +0.213], clearing on 11 of 11 targets.
  It is the widest-clearing result the project has produced -- and it is ONE LAYER.

  F-76 is the standing warning: the kd matched-slice claim looked like a 16-layer plateau on
  point estimates and did not survive resampling, and its argmax was unidentified across 93% of
  the network. A single-layer number is not a result in this project any more, however large.

  PREDICTION, written before the run. Pooling should be MORE layer-robust than any persona
  result, because it removes dataset-specific structure rather than relying on a particular
  direction being well-formed at a particular depth. Concretely: the margin over single-source
  transfer should stay positive at every layer above roughly L12, and the early layers (L0-L8)
  should be weakly positive rather than negative, since both conditions degrade together there.
  If instead the margin is confined to a band around L40, then F-80 is a layer artifact and must
  be reported the way F-76 reports F-03.

  CPU only. Writes to run-premise/, which holds nothing else.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "premise_layers.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^DONE' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# every layer passB stores
$LAYERS = @(0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76)

foreach ($L in $LAYERS) {
    $outF = Join-Path $repo "run-premise\premise_L${L}_mean_s42.json"
    if ((Test-Path $outF) -and -not $Force) { Note "L$L already present, skipping"; continue }
    $t0 = Get-Date
    & $py (Join-Path $D "30_premise.py") --datasets @DS --layer $L --out_root "run-premise" `
        *>> (Join-Path $logDir "premise_layers.log")
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    # Gate on the NUMBER BEING PRESENT, not on the file existing: a run that wrote a file
    # without the leave-one-out column would be the silent failure worth catching, since the
    # LODO rows are the only ones comparable to single-source transfer.
    $ok = $false; $lodo = "n/a"; $single = "n/a"
    if (Test-Path $outF) {
        try {
            $j = Get-Content $outF -Raw | ConvertFrom-Json
            $lodo = [math]::Round($j.summary.pooled_lodo, 4)
            $single = [math]::Round($j.summary.single, 4)
            $ok = ($null -ne $j.summary.pooled_lodo) -and ($null -ne $j.summary.single)
        } catch { $ok = $false }
    }
    Note ("L{0,-3} exit {1}  lodo={2}  single={3}  diff={4}  {5} min  {6}" -f
          $L, $LASTEXITCODE, $lodo, $single,
          $(if ($ok) { [math]::Round($lodo - $single, 4) } else { "n/a" }),
          $mins, $(if ($ok) { "PASS" } else { "GATE FAILED" }))
}

$n = @(Get-ChildItem (Join-Path $repo "run-premise") -Filter "premise_L*_mean_s42.json" -EA SilentlyContinue).Count
Note ("layers with a result: {0} (predicted {1})  {2}" -f $n, $LAYERS.Count,
      $(if ($n -ge $LAYERS.Count) { "PASS" } else { "GATE FAILED" }))

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "premise_layers/final" -EA SilentlyContinue
Note "DONE"
