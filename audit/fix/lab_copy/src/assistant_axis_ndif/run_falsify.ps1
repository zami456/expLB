<#
  The F-86 falsification sweep, plus the layer/seed robustness F-86 never had.

  TWO REVIEW WEAKNESSES, ONE RUN. `docs/REVIEW_REQUEST_POOLING.md` lists the mean-of-directions
  result as the part of the project most likely to be wrong, for two separate reasons:

    q4  "if the average of ten *random* directions with matched norms transferred comparably,
         the story dies. I have not run that."
    w4  "Result 2 is one layer, one seed. Result 1 has 21 layers and five seeds; the mechanism
         does not, and it is the more surprising claim."

  30e_falsify_meandir.py answers q4 with a three-null battery (isotropic, covariance-matched,
  and label-shuffled probes). Sweeping it over layers and seeds answers w4 with the same runs,
  because every invocation re-reports `real`.

  GATES ASSERT NUMBERS. The L40/s42 cell must reproduce F-86's 0.883 within 0.005 -- if it does
  not, the splits have drifted and every comparison in the sweep is against the wrong baseline,
  so the run is void rather than merely disagreeing. Three gates today passed while measuring
  nothing; this one reads back the quantity it names.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

# Do NOT introduce a variable named $d or $src below. PowerShell variable names are
# CASE-INSENSITIVE: $d silently overwrote $D on 2026-09-21 16:00 and $src overwrote $SRC at
# 20:00, each time failing every later step in the same second. Twice in one day.
$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "falsify.status"
$log = Join-Path $logDir "falsify.log"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+DONE\s*$' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# The 21 layers actually stored in passB. NOT a round-number guess: the first run of this
# script asked for layer 30 and three cells died instantly with "layer 30 not stored in
# chunk_0000.pt". Read the store, do not assume a stride.
$LAYERS = @(0, 4, 8, 12, 16, 20, 22, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76)
$SEEDS = @(42, 7, 13)
$total = $LAYERS.Count * $SEEDS.Count
$done = 0

foreach ($L in $LAYERS) {
    foreach ($sd in $SEEDS) {
        $t0 = Get-Date
        $outRoot = Join-Path $repo "run-premise\falsify"
        $dest = Join-Path $outRoot "falsify_L${L}_s${sd}.json"
        if ((Test-Path $dest) -and -not $Force) {
            Note "L$L s$sd  exists, skipping"; $done++; continue
        }
        Note "L$L s$sd  start ($($done + 1)/$total)"
        & $py (Join-Path $D "30e_falsify_meandir.py") --datasets @DS --layer $L --seed $sd `
            --reps 20 --out_root $outRoot *>> $log
        $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
        if (Test-Path $dest) {
            $j = Get-Content $dest -Raw | ConvertFrom-Json
            $worst = ($j.gauss.oriented, $j.cov.oriented, $j.shuffled.oriented |
                      Measure-Object -Maximum).Maximum
            Note ("L$L s$sd  real={0:N4}  best-null={1:N4}  gap={2:N4}  budget10={3:N4}  {4}m" -f `
                  $j.real, $worst, ($j.real - $worst), $j.budget10, $mins)
            $done++
        } else {
            Note "L$L s$sd  NO OUTPUT after ${mins}m -- see falsify.log"
        }
    }
}

# ---- gates: the reproduction check first, then the falsification verdict -----------------
$ref = Join-Path $repo "run-premise\falsify\falsify_L40_s42.json"
if (Test-Path $ref) {
    $j = Get-Content $ref -Raw | ConvertFrom-Json
    $drift = [math]::Abs($j.real - 0.883)
    Note ("GATE reproduce F-86  measured real={0:N4}  predicted=0.883+-0.005  {1}" -f `
          $j.real, $(if ($drift -le 0.005) { "PASS" } else { "GATE FAILED -- splits drifted, sweep is void" }))
} else {
    Note "GATE reproduce F-86  measured=<missing>  GATE FAILED"
}

$all = @(Get-ChildItem (Join-Path $repo "run-premise\falsify\falsify_L*_s*.json") -EA SilentlyContinue |
         ForEach-Object { Get-Content $_.FullName -Raw | ConvertFrom-Json })
if ($all.Count) {
    # GATE ON THE LAYERS WHERE THE METHOD IS CLAIMED TO WORK. The first version took the minimum
    # gap over ALL cells and so logged "min gap = -0.0221 FALSIFIED" on 2026-09-21 -- that
    # minimum is layer 0, token embeddings, where no deception direction should exist and where
    # a PASS would mean the null was broken. Failing there is the calibration working, and a
    # gate that reads it as falsification is measuring the wrong thing. Shallow layers are now
    # reported separately as the calibration check they are. Fourth mis-specified gate today.
    $SHALLOW = 16
    $gapOf = {
        param($c)
        $w = ($c.gauss.oriented, $c.cov.oriented, $c.shuffled.oriented | Measure-Object -Maximum).Maximum
        $c.real - $w }
    $deepCells = @($all | Where-Object { $_.layer -ge $SHALLOW })
    $shallowCells = @($all | Where-Object { $_.layer -lt $SHALLOW })
    $gaps = @($deepCells | ForEach-Object { & $gapOf $_ })
    $minGap = if ($gaps.Count) { ($gaps | Measure-Object -Minimum).Minimum } else { [double]::NaN }
    $shClear = @($shallowCells | Where-Object { (& $gapOf $_) -gt 0.02 }).Count
    Note ("GATE calibration    measured {0}/{1} cells BELOW L{2} clear  predicted few -- failing here is correct" -f `
          $shClear, $shallowCells.Count, $SHALLOW)
    $reals = @($all | ForEach-Object { $_.real })
    $rmin = ($reals | Measure-Object -Minimum).Minimum
    $rmax = ($reals | Measure-Object -Maximum).Maximum
    Note ("GATE falsification  measured min gap over {0} cells at L>={1} = {2:N4}  predicted >0.02  {3}" -f `
          $deepCells.Count, $SHALLOW, $minGap, $(if ($minGap -gt 0.02) { "SURVIVES" } else { "FALSIFIED" }))
    $nlay = @($all | ForEach-Object { $_.layer } | Sort-Object -Unique).Count
    $pos = @($gaps | Where-Object { $_ -gt 0.02 }).Count
    Note ("GATE stability      measured real range {0:N4}-{1:N4} over {2} cells / {3} layers" -f `
          $rmin, $rmax, $all.Count, $nlay)
    Note ("GATE coverage       measured {0}/{1} cells clear the best null by >0.02" -f `
          $pos, $all.Count)
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $repo "kd\scripts\sync_kd.ps1") `
    -OnBusy Wait -Tag "falsify/final" -EA SilentlyContinue
Note "DONE"
