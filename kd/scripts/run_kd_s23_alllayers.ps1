<#
  NEXT.md item 3: extend the kd_s23 crossgen grid from 3 layers to all 21 stored in passB.

  WHY. Every headline the kd bank has produced is read at L40, with L20/L44 as the only
  comparison points. F-06 already showed the layer choice is not cosmetic -- knowledge and
  wording are entangled at L40 (cos 0.410) and nearly orthogonal at L20 (0.063) -- and F-17's
  Q6 result is that the right layer depends on the method. Three layers cannot distinguish "the
  effect peaks at 40" from "the effect exists only at 40", and those mean different things.

  NO NDIF. This reads activations already stored in run-coords/passB and axes already built in
  run-kd/axes_vectors_s23.pt. It is CPU + disk only, so it can run beside the scale rungs
  without competing for the API.

  CHEAP BY CONSTRUCTION. `27_crossgen.py` skips a matrix whose file already exists
  ("skip mean_L40_s42.json (exists)"), and `--layers` with no values defaults to every stored
  layer. So omitting --layers entirely computes exactly the 18 that are missing and re-reads
  none of the 3 that are done. Passing an explicit list would risk typing a layer passB does
  not hold.

  The feature list, dataset list and bank are copied verbatim from run_kd_s23.ps1 lines 84-99
  so the new layers are directly comparable to the three already measured. Changing any of
  them here would make the 21-layer sweep a different experiment from its own first 3 points.

  $SRC / $step naming is deliberate -- see run_kd_s23.ps1's header. PowerShell variables are
  case-insensitive and `$s` clobbering `$S` cost two crossgen steps once already.
#>
param([switch]$WhatIf)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$out = Join-Path $repo "run-kd"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "kd_s23_alllayers.status"
$bank = Join-Path $out "axes_vectors_s23.pt"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Append -Encoding utf8 $status

# Preflight. A missing bank or run_dir fails 21 layers deep otherwise, after an hour.
foreach ($p in @($bank, (Join-Path $repo "run-coords\passB"), $pyAN)) {
    if (-not (Test-Path $p)) { Note "MISSING $p -- aborting"; exit 1 }
}
$existing = @(Get-ChildItem (Join-Path $repo "run-crossgen\kd11\banks\kd_s23\matrices") `
                -Filter "mean_L*_s42.json" -EA SilentlyContinue).Count
Note "preflight ok; $existing mean-pooled matrices already present (expect 3)"

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# No --layers: every stored layer. Existing matrices are skipped by the script itself.
$a = @("$SRC\27_crossgen.py", "--run_dir", "run-coords\passB",
       "--out_root", "run-crossgen\kd11\banks\kd_s23", "--bank_name", "kd_s23",
       "--bank", $bank, "--baseline_root", "run-crossgen\kd11",
       "--datasets") + $DS + @(
       "--features", "pc:deception", "axes", "axes_resid", "pc:unified",
                     "pc:knowledge_denial", "pc:kd_assertion",
       "--extra_features", "axes", "pc:knowledge_denial", "pc:kd_assertion",
       "--stage", "all")

if ($WhatIf) { Note "WHATIF: $pyAN $($a -join ' ')"; "would run:"; "$pyAN $($a -join ' ')"; exit 0 }

Note "crossgen s23 ALL layers (no --layers flag) start"
& $pyAN @a 2>&1 | ForEach-Object { "$_" } |
    Out-File -Append -Encoding utf8 (Join-Path $logDir "kd_s23_alllayers.log")
Note "crossgen s23 all-layers exit $LASTEXITCODE"

$after = @(Get-ChildItem (Join-Path $repo "run-crossgen\kd11\banks\kd_s23\matrices") `
             -Filter "mean_L*_s42.json" -EA SilentlyContinue).Count
Note "mean-pooled matrices now: $after (was $existing)"

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") `
    -OnBusy Wait -Tag "s23alllayers/final"
Note "DONE"
