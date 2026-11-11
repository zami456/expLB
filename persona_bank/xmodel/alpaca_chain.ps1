<#
  Qwen's Liars' Bench Alpaca rows, mean-pooled over the reply at all 80 layers (owner, 2026-10-07: "run the
  Alpaca eval read for D9 + K"). The selected detector D9 + K divides each direction by the spread of its
  projection on Alpaca (zami's recall_denial.standardised_sum); this read supplies that control set.

  It is the run_chain eval stage restricted to alpaca: 8_coords_ndif.py --datasets alpaca --layers all
  --pooling mean --span response, the model's own LB rows (XMODEL_CSV_MODEL; without it eval_datasets loads
  Llama's rows -- seen 2026-10-07 14:13), the laptop download floor passed as arguments (8_coords installs its
  own floor, not ndif_common.harden's).

  Gate: xmodel_status.py's eval count (rows pooled == rows in items.pt), never an exit code -- 8_coords exits 0
  after a session fails permanently and leaves its rows for the next attempt.
  Outputs: <Run>\eval\alpaca\ (chunks; inside the run folder, copied to Drive at the end).
  Live logs: <LogDir> (outside the synced folder). Status: <Run>\logs\alpaca.status.
  Resumable; exits at once after DONE. Scheduled Task XModel-qwen-alpaca (register_alpaca_task.ps1).
#>
param([string]$Run = "$env:USERPROFILE\xmodel-runs\Qwen2.5-72B-Instruct",
      [string]$LogDir = "$env:USERPROFILE\xmodel-runs\alpaca-logs",
      [string]$Model = "Qwen/Qwen2.5-72B-Instruct",
      [string]$CsvModel = "qwen-2.5-72b-instruct",
      [string]$DriveBase = "tamara:expLB/xmodel_2026-10-06",
      [string]$Rclone = "C:\Users\SURAIYA\AppData\Local\Microsoft\WinGet\Packages\Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe\rclone-v1.75.1-windows-amd64\rclone.exe",
      [int]$Attempts = 8,
      # 2026-10-08 (Gemma-3): its outlier dims overflow fp16 at layers 45-61, so its read is redone in float32 into
      # <Run>\eval_fp32\alpaca (the fp16 store in <Run>\eval\alpaca is kept; its layers 0-44 are finite).
      # Defaults leave every earlier read exactly as it was.
      [string]$EvalDir = "eval", [ValidateSet("float16", "float32")][string]$StoreDtype = "float16",
      [string]$StatusName = "alpaca.status",
      # 2026-10-09 (Llama-405B, 126 x 16384): a layer subset, space-separated ints, instead of all layers
      [string]$Layers = "all")
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo ".venv-aa-win\Scripts\python.exe"
$code = Join-Path $Run "code"
$S = Join-Path $code "src\assistant_axis_ndif"
New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Run "logs") | Out-Null
$status = Join-Path $Run "logs\$StatusName"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Counts {
    if ($EvalDir -eq "eval") {
        $j = & $py "$S\xmodel_status.py" --root $Run --json 2>$null
        try { return (($j -join "`n") | ConvertFrom-Json).eval.alpaca } catch { return $null }
    }
    # any other store: the same counting function, pointed at that folder
    $c = "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import xmodel_status as X; print(json.dumps(X.eval_counts(Path(sys.argv[2]))))"
    $j = & $py -c $c $S (Join-Path $Run $EvalDir) 2>$null
    try { return (($j -join "`n") | ConvertFrom-Json).alpaca } catch { return $null }
}

if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match '  DONE')) { exit 0 }
$env:XMODEL_CSV_MODEL = $CsvModel
$env:PYTHONIOENCODING = "utf8"
# ndif_common.harden() installs its own floor at import (300 KB/s by default); 8_coords' --min_download_kbps
# replaces it, since both assign RemoteBackend.get_result, but keep the two in agreement
$env:NDIF_MIN_DOWNLOAD_KBPS = "40"
$env:NDIF_DOWNLOAD_GRACE = "20"
Note "START model=$Model csv=$CsvModel run=$Run py=$py"

$a = @("`"$S\8_coords_ndif.py`"", "--model", $Model, "--datasets", "alpaca", "--output_dir", "`"$Run\$EvalDir`"",
       "--layers") + @($Layers -split '\s+' | Where-Object { $_ }) + @("--pooling", "mean", "--span", "response",
       "--min_download_kbps", "40", "--download_grace", "20")
# only when asked: run folders deployed before 2026-10-08 have an 8_coords without --store_dtype
if ($StoreDtype -ne "float16") { $a += @("--store_dtype", $StoreDtype) }
$complete = $false
for ($t = 1; $t -le $Attempts; $t++) {
    $c = Counts
    if ($c -and $c.items -and $c.done -ge $c.items) { $complete = $true; break }
    Note "read attempt $t (have $($c.done) / $($c.items) rows)"
    $out = Join-Path $LogDir "alpaca.$t.log"
    $err = Join-Path $LogDir "alpaca.$t.err.log"
    $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $code -NoNewWindow -PassThru `
            -RedirectStandardOutput $out -RedirectStandardError $err
    $null = $p.Handle   # PS 5.1 reports ExitCode only if the handle was taken before exit
    $p.WaitForExit()
    Note "read attempt $t exit $($p.ExitCode): $(Get-Content $err -Tail 1 -ErrorAction SilentlyContinue)"
    $c = Counts
    if ($c -and $c.items -and $c.done -ge $c.items) { $complete = $true; break }
    Start-Sleep -Seconds 120
}
$c = Counts
if (-not $complete) { Note "STOP alpaca $($c.done)/$($c.items) rows after $Attempts attempts (a re-run resumes)"; exit 1 }
Note "alpaca: $($c.done)/$($c.items) rows, row shape $($c.row_shape -join 'x')"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync_drive.ps1") `
    -Folder (Split-Path $Run -Leaf) -RunBase (Split-Path $Run -Parent) -DriveBase $DriveBase -Rclone $Rclone `
    -Tag "alpaca" -SkipLiveLogs
Note "drive copy exit $LASTEXITCODE"
Note "DONE"
