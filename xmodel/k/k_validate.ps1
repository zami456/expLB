<#
  Validation of the K setup (owner, 2026-10-07): zami's code, through k_run.py and .venv-k, on Llama-3.3-70B,
  compared with zami's committed results (results/recall_denial: verify.json, split_half.csv none|general at
  layers 20/36/40/44). If this reproduces zami's numbers, the environment and driver are faithful, and the
  Qwen K built with them can be trusted.

  Waits until the Qwen K chain is DONE (one job at a time on the laptop link; Llama-3.3 may need a cold deploy).
  Outputs: <Out>\llama-70b-3.3\first\ (outside the Qwen run folder). Status: <Out>\validate.status.
  The comparison is written to <Out>\compare.txt. Scheduled Task XModel-K-validate (register_k_validate.ps1).
#>
param([string]$Out = "$env:USERPROFILE\xmodel-runs\K-validation-llama",
      [string]$QwenK = "$env:USERPROFILE\xmodel-runs\Qwen2.5-72B-Instruct\K\k_chain.status",
      [int]$Attempts = 4)
$ErrorActionPreference = "Continue"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$py = Join-Path $repo ".venv-k\Scripts\python.exe"
$drv = Join-Path $PSScriptRoot "k_run.py"
$logs = Join-Path $Out "logs"
New-Item -ItemType Directory -Force -Path $Out, $logs | Out-Null
$status = Join-Path $Out "validate.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match '  DONE')) { exit 0 }
$said = $false
while (-not ((Test-Path $QwenK) -and ((Get-Content $QwenK -Tail 1) -match '  DONE'))) {
    if (-not $said) { Note "waiting for the Qwen K chain to finish"; $said = $true }
    Start-Sleep -Seconds 120
}
$env:K_OUT_ROOT = $Out
$env:PYTHONIOENCODING = "utf8"
Note "START Llama-3.3 validation of the K setup (py=$py)"

foreach ($step in "verify", "items", "read", "directions", "compare") {
    $a = @("`"$drv`"", $step, "--model", "llama-70b-3.3", "--set", "first", "--layers", "zami")
    $ok = $false
    for ($t = 1; $t -le $Attempts; $t++) {
        $o = Join-Path $logs "v_${step}.$t.log"
        $e = Join-Path $logs "v_${step}.$t.err.log"
        $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
                -RedirectStandardOutput $o -RedirectStandardError $e
        $null = $p.Handle
        $p.WaitForExit()
        if ($p.ExitCode -eq 0) { $ok = $true; break }
        Note "$step attempt $t exit $($p.ExitCode): $(Get-Content $e -Tail 1 -ErrorAction SilentlyContinue)"
        Start-Sleep -Seconds 300
    }
    if (-not $ok) { Note "STOP $step failed $Attempts times (a re-run resumes)"; exit 1 }
    if ($step -eq "compare") { Copy-Item $o (Join-Path $Out "compare.txt") -Force }
    Note "$step ok"
}
Note "DONE"
