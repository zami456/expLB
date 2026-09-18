<#
  Runs the 10 x 10 cross-generalization retrain (27_crossgen.py) on the lab PC, then copies
  run-crossgen/ to Drive. Started by the Crossgen-Retrain scheduled task, never over plain SSH
  (Windows OpenSSH takes the process tree down with the session).

  Everything lands in run-crossgen/ and tamara:expLB/crossgen -- never run-coords/.

      -Smoke                 layer 44, last pooling, seed 42, plus the controls
      -ExtraSeeds "0 1 2 3 4" additional 80/20 splits for error bands
      -ExtraLayers "20 22 28 32 40 44 48"   restrict the extra seeds to these layers
#>
param(
    [switch]$Smoke,
    [string]$ExtraSeeds = "",
    [string]$ExtraLayers = ""
)
$ErrorActionPreference = "Continue"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$tag = if ($Smoke) { "crossgen_smoke" } else { "crossgen" }
$status = Join-Path $repo "logs\$tag.status"
$log = Join-Path $repo "logs\$tag.log"

"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$script = Join-Path $PSScriptRoot "27_crossgen.py"
$pyArgs = @($script)
if ($Smoke) {
    $pyArgs += "--smoke"
} else {
    $pyArgs += @("--stage", "all")
    if ($ExtraSeeds.Trim()) { $pyArgs += @("--extra_seeds") + ($ExtraSeeds.Trim() -split "\s+") }
    if ($ExtraLayers.Trim()) { $pyArgs += @("--extra_layers") + ($ExtraLayers.Trim() -split "\s+") }
}

& $py @pyArgs 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Encoding utf8
$code = $LASTEXITCODE
"PYTHON_EXIT $code " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status

# Drive copy -- rclone is not on PATH under the S4U task, so address it by path.
$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if ((Test-Path $rc) -and (Test-Path (Join-Path $repo "run-crossgen"))) {
    & $rc copy (Join-Path $repo "run-crossgen") "tamara:expLB/crossgen" --checksum --transfers 4 `
        --log-file (Join-Path $repo "logs\${tag}_rclone.log") --log-level INFO
    "RCLONE_EXIT $LASTEXITCODE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
} else {
    "RCLONE_SKIPPED " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
