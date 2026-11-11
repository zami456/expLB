<#
  Copy one random-persona-bank run folder to Drive: <DriveBase>/<Folder>
  (default tamara:expLB/random_persona_bank_2026-10-10/<Folder>).

  A copy of xmodel/sync_drive.ps1 (2026-10-10); only this header and the RunBase / DriveBase defaults differ.
  A NEW dated folder, created by this run; `rclone copy` only (never sync/move/delete), so
  nothing already on Drive is touched. The .env holding the NDIF/HF keys is excluded -- it is
  in code\ only so the pipeline can authenticate on this machine.

  Exclusive file lock: two rclone processes writing one Drive path leave duplicate names.
#>
param([Parameter(Mandatory)][string]$Folder, [string]$Tag = "sync", [int]$WaitSeconds = 3600,
      # empty RunBase = the results folder this copy was deployed under (<RunBase>\<Folder>\code\sync_drive.ps1)
      [string]$RunBase = "", [string]$DriveBase = "tamara:expLB/random_persona_bank_2026-10-10",
      [string]$Rclone = "",
      # periodic copies while a stage runs: skip the per-process stdout logs (<stage>.<n>.log),
      # which grow during the upload and fail Drive's md5 check (seen 2026-10-06 23:20)
      [switch]$SkipLiveLogs)
$ErrorActionPreference = "Continue"
if (-not $RunBase) { $RunBase = Split-Path (Split-Path (Split-Path $PSCommandPath -Parent) -Parent) -Parent }
$Root = "$RunBase\$Folder"
$Logs = "$Root\logs"
$status = "$Logs\sync.status"
New-Item -ItemType Directory -Force -Path $Logs | Out-Null
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

$rc = $Rclone
if (-not $rc -or -not (Test-Path $rc)) { $c = Get-Command rclone -ErrorAction SilentlyContinue; if ($c) { $rc = $c.Source } }
if (-not $rc -or -not (Test-Path $rc)) { Note "$Tag NO RCLONE"; exit 1 }
$dest = "$DriveBase/$Folder"

$lock = "$Logs\rclone.lock"
$deadline = (Get-Date).AddSeconds($WaitSeconds)
$fs = $null
while ($true) {
    try { $fs = [System.IO.File]::Open($lock, 'OpenOrCreate', 'ReadWrite', 'None'); break }
    catch { if ((Get-Date) -gt $deadline) { Note "$Tag GAVE UP waiting for lock"; exit 1 }; Start-Sleep -Seconds 15 }
}
try {
    Note "$Tag START -> $dest"
    # logs/rclone.log is excluded always: rclone writes it during this very copy, so it can never
    # match its own md5 (2026-10-06).
    $live = @()
    if ($SkipLiveLogs) { $live = @("--exclude", "logs/*.[0-9].log", "--exclude", "logs/*.[0-9].err.log") }
    & $rc copy $Root $dest --checksum --transfers 4 `
        --exclude "**/.env" --exclude "**/__pycache__/**" --exclude "logs/rclone.lock" `
        --exclude "logs/rclone.log" @live `
        --log-file "$Logs\rclone.log" --log-level INFO
    Note "$Tag END exit $LASTEXITCODE"
} finally { if ($fs) { $fs.Close() } }
