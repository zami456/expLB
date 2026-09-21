<#
  The three CPU items still owed on the premise result, in one chain.

  1. SEED ROBUSTNESS. F-81 holds across 21 layers but on one split seed. Twenty-one agreeing
     layers make a seed artifact unlikely, but "unlikely" is not "measured". Re-runs L40 at
     seeds 1, 7, 13, 99 and reports the spread of the margin.

  2. WITHIN-CLASS MAHALANOBIS. F-79 reported R^2 = 0.017 for Mahalanobis cosine against Ying et
     al.'s 0.98 -- but used the TOTAL covariance where Mahalanobis normally means the pooled
     WITHIN-CLASS scatter. That was our implementation error, not their failure to replicate, and
     F-79 says so. 30_premise.py now defaults to within-class; this re-runs L40 both ways so the
     two are side by side.

  3. CLUSTER-BASED PERMUTATION TEST on the kd matched-slice layer profile -- the one test
     docs/LAYER_PREREGISTRATION.md committed to that has never run. It tests a different null
     from the dataset bootstrap (labels, not datasets) and needs probe refits, which is why it
     was deferred. Run by 30c_cluster_perm.py.

  CPU only. Everything writes under run-premise/.
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
$status = Join-Path $logDir "premise_extras.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^DONE' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# ---- 1. seeds -------------------------------------------------------------------------
foreach ($sd in @(1, 7, 13, 99)) {
    $t0 = Get-Date
    & $py (Join-Path $D "30_premise.py") --datasets @DS --layer 40 --seed $sd `
        --out_root "run-premise" *>> (Join-Path $logDir "premise_extras.log")
    $f = Join-Path $repo "run-premise\premise_L40_mean_s$sd.json"
    $d = "n/a"
    if (Test-Path $f) {
        try {
            $j = Get-Content $f -Raw | ConvertFrom-Json
            $d = [math]::Round($j.summary.pooled_lodo - $j.summary.single, 4)
        } catch {}
    }
    Note ("seed {0}  exit {1}  lodo-single={2}  {3} min  {4}" -f $sd, $LASTEXITCODE, $d,
          [math]::Round(((Get-Date) - $t0).TotalMinutes, 1),
          $(if ($d -ne "n/a") { "PASS" } else { "GATE FAILED" }))
}

# ---- 2. Mahalanobis, both metrics ------------------------------------------------------
foreach ($cov in @("within", "total")) {
    & $py (Join-Path $D "30_premise.py") --datasets @DS --layer 40 --seed 42 `
        --covariance $cov --out_root "run-premise" *>> (Join-Path $logDir "premise_extras.log")
    $tag = if ($cov -eq "within") { "" } else { "_$cov" }
    $f = Join-Path $repo "run-premise\premise_L40_mean_s42$tag.json"
    $r2 = "n/a"
    if (Test-Path $f) {
        try {
            $j = Get-Content $f -Raw | ConvertFrom-Json
            $r2 = [math]::Round($j.predictors.mahal_cos.r2, 4)
        } catch {}
    }
    Note ("mahalanobis/{0}  exit {1}  R2={2}  (Ying et al. report 0.98)  {3}" -f
          $cov, $LASTEXITCODE, $r2, $(if ($r2 -ne "n/a") { "PASS" } else { "GATE FAILED" }))
}

# ---- 3. cluster permutation ------------------------------------------------------------
$t0 = Get-Date
& $py (Join-Path $D "30c_cluster_perm.py") *>> (Join-Path $logDir "premise_extras.log")
Note ("cluster_perm exit {0}  {1} min" -f $LASTEXITCODE,
      [math]::Round(((Get-Date) - $t0).TotalMinutes, 1))

Note "DONE"
