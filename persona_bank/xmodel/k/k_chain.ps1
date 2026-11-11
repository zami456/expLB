<#
  Qwen K (zami's knowledge direction, report sec. 4.2.8) beside the persona chain (owner, 2026-10-07).

  For each question set (first = recall_items, matched = recall_items_matched):
      verify -> items -> read (all 80 layers) -> directions
  k_run.py --gate_status makes every NDIF request wait until the persona chain is DONE (2026-10-07 02:56:
  in the judge window K's ~52 MB reads starved the judge on this laptop's link -- WriteTimeout, 3 KB/s).

  Outputs: <OutRoot>\qwen-72b\<set>\ (inside the run folder, so XModel-qwen-sync copies them to Drive).
  Live logs: <LogDir> (outside the synced folder: a log growing mid-upload fails Drive's md5 check).
  Status: <OutRoot>\k_chain.status. Resumable and idempotent; exits at once after DONE.
  Scheduled Task XModel-qwen-K (register_k_task.ps1).
#>
param([string]$Model = "qwen-72b",
      [string]$OutRoot = "$env:USERPROFILE\xmodel-runs\Qwen2.5-72B-Instruct\K",
      [string]$LogDir = "$env:USERPROFILE\xmodel-runs\K-logs",
      [string]$Gate = "$env:USERPROFILE\xmodel-runs\Qwen2.5-72B-Instruct\logs\chain.status",
      [int]$Attempts = 4,
      # padded tokens per read request (2026-10-09): 2048 returns ~240 MB per request at 126 x 16384 (Llama-405B), so
      # that run passes 512; every earlier run used 2048, the default
      [int]$TokenBudget = 2048)
$ErrorActionPreference = "Continue"
$repo = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$py = Join-Path $repo ".venv-k\Scripts\python.exe"
$drv = Join-Path $PSScriptRoot "k_run.py"
New-Item -ItemType Directory -Force -Path $OutRoot, $LogDir | Out-Null
$status = Join-Path $OutRoot "k_chain.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match '  DONE')) { exit 0 }
$env:K_OUT_ROOT = $OutRoot
$env:PYTHONIOENCODING = "utf8"
Note "START model=$Model out=$OutRoot gate=$Gate py=$py"

foreach ($set in "first", "matched") {
    foreach ($step in "verify", "items", "read", "directions") {
        $a = @("`"$drv`"", $step, "--model", $Model, "--set", $set, "--layers", "all",
               "--token_budget", "$TokenBudget", "--gate_status", "`"$Gate`"")
        $ok = $false
        for ($t = 1; $t -le $Attempts; $t++) {
            $out = Join-Path $LogDir "k_${set}_${step}.$t.log"
            $err = Join-Path $LogDir "k_${set}_${step}.$t.err.log"
            $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
                    -RedirectStandardOutput $out -RedirectStandardError $err
            $null = $p.Handle   # PS 5.1 reports ExitCode only if the handle was taken before exit
            $p.WaitForExit()
            if ($p.ExitCode -eq 0) { $ok = $true; break }
            $last = (Get-Content $err -Tail 1 -ErrorAction SilentlyContinue)
            Note "$set $step attempt $t exit $($p.ExitCode): $last"
            Start-Sleep -Seconds 300
        }
        if (-not $ok) { Note "STOP $set $step failed $Attempts times (a re-run resumes)"; exit 1 }
        Note "$set $step ok"
    }
}
Note "DONE"
