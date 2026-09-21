<#
  Phase 5: score the 11-dataset grid with the NEW 36-persona kd bank.

  Waits for Phase 3 to produce run-kd\axes_vectors_s3.pt, then runs 27_crossgen.py with that
  bank against the SAME splits and baselines as the root run in run-crossgen\kd11, so every
  comparison is cell by cell on identical rows. Persona features only -- raw, random, dataset
  PCA and length are merged from the root at summarize time via --baseline_root.

  Features include the two contrasts this whole experiment exists to test. They come from the
  bank's own `contrasts` block (6_axes.py reads it), so they are available as pc:<name>:
      pc:knowledge_denial   feigned ignorance - SINCERE ignorance (both sides deny;
                            cannot be a wording artifact)
      pc:kd_assertion       feigned ignorance - plain assertion (the register contrast)

  Chained rather than launched by hand so the night is not spent waiting on a human. Every
  stage is resumable; re-entry after a reboot is free.
#>
param([int]$MaxWaitHours = 8, [int]$PollSeconds = 120)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$script = Join-Path $repo "src\assistant_axis_ndif\27_crossgen.py"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "kd_bank_crossgen.status"
$log = Join-Path $logDir "kd_bank_crossgen.log"
$bank = Join-Path $repo "run-kd\axes_vectors_s3.pt"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

Note "WAITING for $bank"
$deadline = (Get-Date).AddHours($MaxWaitHours)
while (-not (Test-Path $bank)) {
    if ((Get-Date) -gt $deadline) { Note "GAVE UP after $MaxWaitHours h -- Phase 3 never produced the bank"; exit 2 }
    Start-Sleep -Seconds $PollSeconds
}
# the chain writes axes last, but give the file a moment to settle before torch.load
Start-Sleep -Seconds 30
Note "bank present ($([math]::Round((Get-Item $bank).Length/1MB,1)) MB)"

# layers in the order that matters: 40 and 44 first (where every conclusion is read and where
# the persona peak sits), then 20 (Apollo's layer, and where the factor probes are nearly
# orthogonal: cos(dK,dU)=0.063 vs 0.41 at L40), then the rest for the layer profile.
$steps = @(
    @{ tag = "peak"; layers = @("40", "44") },
    @{ tag = "l20";  layers = @("20") },
    @{ tag = "rest"; layers = @("0","4","8","12","16","22","24","28","32","36","48","52",
                                "56","60","64","68","72","76") }
)
foreach ($s in $steps) {
    Note "STEP $($s.tag) layers=$($s.layers -join ',') start"
    $a = @("$script", "--run_dir", "run-coords\passB",
           "--out_root", "run-crossgen\kd11\banks\kd", "--bank_name", "kd",
           "--bank", $bank, "--baseline_root", "run-crossgen\kd11",
           "--datasets") + $DS + @("--layers") + $s.layers + @(
           "--features", "pc:deception", "axes", "axes_resid", "pc:unified", "pc:unified@2",
                         "pc:knowledge_denial", "pc:kd_assertion",
           "--extra_features", "axes", "pc:knowledge_denial", "pc:kd_assertion",
           "--stage", "all")
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    Note "STEP $($s.tag) exit $LASTEXITCODE"
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") `
        -OnBusy Skip -Tag "bankcrossgen/$($s.tag)"
}
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") -OnBusy Wait -Tag "bankcrossgen/final"
Note "DONE"
