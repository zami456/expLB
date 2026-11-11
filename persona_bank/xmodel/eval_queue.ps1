<#
  Liars' Bench eval reads for the second models -- the "other side of the dot product" (owner 2026-10-09: "detection
  numbers"). For each model in turn (Mistral-3.1, Qwen, Gemma-3), after -WaitFor ends in DONE:
    0 sync     the repo's current pipeline files into the run folder's code (byte copies, md5 logged): eval_datasets has the
               per-model soft-trigger / gender-secret organisms, 8_coords the Gemma span fix and --store_dtype, check_spans
               and xmodel_status as used everywhere else. Liars' Bench CSVs copied from results/subsets when missing.
    1 spans    8_coords --prepare_only, then check_spans per dataset; a dataset is read only if its spans pass
    2 read     8_coords, mean over the reply, every 4th layer + half depth, into <Run>\eval_lb\ (the all-layer Alpaca
               stores in eval\ and eval_fp32\ are left alone); Gemma in float32 (its outlier dims overflow fp16 past L45)
    3 drive    sync_drive.ps1 for the run folder
  Soft-trigger / gender-secret rows were written by Liars' Bench's LoRA organisms; here they are read through the BASE
  model, as the report's soft-trigger "base read" was. Resumable; a model whose reads are complete is skipped.
  Status: %USERPROFILE%\xmodel-runs\eval_queue.status. Scheduled Task XModel-eval-queue (register_eval_queue.ps1).
#>
param([string]$WaitFor = "$env:USERPROFILE\xmodel-runs\Mistral-Small-3.1-24B-Instruct-2503\logs\alpaca_gen.status",
      [string]$DriveBase = "tamara:expLB/xmodel_2026-10-06",
      [string]$Rclone = "~\AppData\Local\Microsoft\WinGet\Packages\Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe\rclone-v1.75.1-windows-amd64\rclone.exe",
      [int]$Attempts = 6,
      # 2026-10-16 (owner: "run Qwen detection"): -Only <tags> reads just those models; -Datasets overrides their dataset
      # lists; -NoDrive skips step 3 (sync_drive copies the WHOLE run folder, and the default DriveBase is an existing folder)
      [string[]]$Only = @(), [string[]]$Datasets = @(), [switch]$NoDrive)
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo ".venv-aa-win\Scripts\python.exe"
$runBase = Join-Path $env:USERPROFILE "xmodel-runs"
$status = Join-Path $runBase "eval_queue.status"
$logDir = Join-Path $runBase "eval-queue-logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match '  ALL DONE')) { exit 0 }
$env:PYTHONIOENCODING = "utf8"
$env:NDIF_MIN_DOWNLOAD_KBPS = "40"
$env:NDIF_DOWNLOAD_GRACE = "20"

function Every4($n, $half) { $l = @(); for ($i = 0; $i -lt $n; $i += 4) { $l += $i }; if ($l -notcontains $half) { $l += $half }; return ($l | Sort-Object) }
$csvOf = @{ "instructed-deception" = "instructed-deception.csv"; "harm-pressure-knowledge-report" = "harm-pressure-knowledge-report.csv"
            "harm-pressure-choice" = "harm-pressure-choice.csv"; "insider-trading" = "insider-trading.csv"
            "convincing-game" = "convincing-game.csv"; "soft-trigger-gender" = "soft-trigger.csv"
            "soft-trigger-greeting" = "soft-trigger.csv"; "soft-trigger-time" = "soft-trigger.csv"; "gender-secret" = "gender-secret.csv" }
$st = @("soft-trigger-gender", "soft-trigger-greeting", "soft-trigger-time")
$models = @(
    @{ Tag = "mistral31"; Folder = "Mistral-Small-3.1-24B-Instruct-2503"; Model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
       Csv = "mistral-small-3.1-24b-instruct"; Layers = (Every4 40 20); Store = "float16"
       Datasets = @("instructed-deception", "harm-pressure-knowledge-report", "harm-pressure-choice", "insider-trading") + $st + @("gender-secret") },
    @{ Tag = "qwen"; Folder = "Qwen2.5-72B-Instruct"; Model = "Qwen/Qwen2.5-72B-Instruct"
       Csv = "qwen-2.5-72b-instruct"; Layers = (Every4 80 40); Store = "float16"
       Datasets = @("instructed-deception", "harm-pressure-knowledge-report", "harm-pressure-choice") + $st },
    @{ Tag = "gemma3"; Folder = "gemma-3-27b-it"; Model = "google/gemma-3-27b-it"
       Csv = "gemma-3-27b-it"; Layers = (Every4 62 31); Store = "float32"
       Datasets = @("instructed-deception", "convincing-game", "insider-trading") + $st }
)

Note "START waitfor=$WaitFor"
if ($WaitFor) {
    $said = $false
    while (-not ((Test-Path $WaitFor) -and ((Get-Content $WaitFor -Tail 1) -match '  DONE$'))) {
        if (-not $said) { Note "waiting for $WaitFor to end in DONE"; $said = $true }
        Start-Sleep -Seconds 120
    }
    Note "gate open"
}

$allComplete = $true
foreach ($m in $models) {
    if ($Only -and ($Only -notcontains $m.Tag)) { continue }
    if ($Datasets) { $m.Datasets = $Datasets }
    $run = Join-Path $runBase $m.Folder
    $code = Join-Path $run "code"
    $S = Join-Path $code "src\assistant_axis_ndif"
    $ev = Join-Path $run "eval_lb"
    $tag = $m.Tag
    if (Test-Path (Join-Path $ev "_DONE")) { Note "$tag skipped (eval_lb\_DONE)"; continue }
    $env:XMODEL_CSV_MODEL = $m.Csv
    # ---- 0 sync code and data ----
    foreach ($f in "8_coords_ndif.py", "ndif_common.py", "eval_datasets.py", "check_spans.py", "xmodel_status.py") {
        Copy-Item (Join-Path $repo "xmodel\code\src\assistant_axis_ndif\$f") (Join-Path $S $f) -Force
    }
    $md5 = (Get-FileHash (Join-Path $S "8_coords_ndif.py") -Algorithm MD5).Hash.Substring(0, 8)
    $sub = Join-Path $code "results\subsets"
    New-Item -ItemType Directory -Force -Path $sub | Out-Null
    foreach ($c in ($m.Datasets | ForEach-Object { $csvOf[$_] } | Select-Object -Unique)) {
        $src = Join-Path $repo "results\subsets\$c"; $dst = Join-Path $sub $c
        if (-not (Test-Path $dst) -or (Get-FileHash $src -Algorithm MD5).Hash -ne (Get-FileHash $dst -Algorithm MD5).Hash) {
            Copy-Item $src $dst -Force; Note "$tag copied $c"
        }
    }
    Note "$tag code synced (8_coords md5 $md5); layers $($m.Layers -join ','); store $($m.Store)"
    # ---- 1 spans ----
    $ok = @()
    foreach ($d in $m.Datasets) {
        if (-not (Test-Path (Join-Path $ev "$d\items.pt"))) {
            $null = & $py "$S\8_coords_ndif.py" --model $m.Model --datasets $d --output_dir $ev --layers all `
                    --pooling mean --span response --prepare_only 2>&1
        }
        $out = & $py "$S\check_spans.py" --model $m.Model --eval_dir $ev --datasets $d 2>&1
        $line = ($out | Where-Object { $_ -match 'spans \(of' }) -join ' '
        if ($LASTEXITCODE -eq 0) { $ok += $d } else { Note "$tag $d SKIPPED, span check failed: $line" }
        Note "$tag spans $d exit ${LASTEXITCODE}: $line"
    }
    if (-not $ok) { Note "$tag no dataset passed the span check"; $allComplete = $false; continue }
    # ---- 2 read ----
    $layerArgs = @($m.Layers | ForEach-Object { "$_" })
    $count = "import json,sys; from pathlib import Path; sys.path.insert(0, sys.argv[1]); import xmodel_status as X; print(json.dumps(X.eval_counts(Path(sys.argv[2]))))"
    $complete = $false
    for ($t = 1; $t -le $Attempts; $t++) {
        $c = $null; try { $c = ((& $py -c $count $S $ev 2>$null) -join "`n") | ConvertFrom-Json } catch {}
        $todo = @($ok | Where-Object { -not ($c -and $c.$_ -and $c.$_.items -and $c.$_.done -ge $c.$_.items) })
        if (-not $todo) { $complete = $true; break }
        Note "$tag read attempt $t ($($todo.Count) datasets to finish: $($todo -join ','))"
        $a = @("`"$S\8_coords_ndif.py`"", "--model", $m.Model, "--datasets") + $todo +
             @("--output_dir", "`"$ev`"", "--layers") + $layerArgs +
             @("--pooling", "mean", "--span", "response", "--min_download_kbps", "40", "--download_grace", "20")
        if ($m.Store -ne "float16") { $a += @("--store_dtype", $m.Store) }
        $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $code -NoNewWindow -PassThru `
                -RedirectStandardOutput (Join-Path $logDir "$tag.$t.log") -RedirectStandardError (Join-Path $logDir "$tag.$t.err.log")
        $null = $p.Handle
        $p.WaitForExit()
        Note "$tag read attempt $t exit $($p.ExitCode)"
        Start-Sleep -Seconds 60
    }
    $c = $null; try { $c = ((& $py -c $count $S $ev 2>$null) -join "`n") | ConvertFrom-Json } catch {}
    foreach ($d in $ok) { if ($c -and $c.$d) { Note "$tag $d $($c.$d.done)/$($c.$d.items) rows, row shape $($c.$d.row_shape -join 'x')" } }
    # ---- 3 drive ----
    if ($NoDrive) { Note "$tag drive copy skipped (-NoDrive)" } else {
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync_drive.ps1") `
        -Folder $m.Folder -RunBase $runBase -DriveBase $DriveBase -Rclone $Rclone -Tag "eval_lb" -SkipLiveLogs
    Note "$tag drive copy exit $LASTEXITCODE" }
    if ($complete) { "done $(Get-Date -Format s)" | Out-File (Join-Path $ev "_DONE"); Note "$tag DONE" }
    else { Note "$tag INCOMPLETE after $Attempts attempts (a re-run resumes)"; $allComplete = $false }
}
if ($allComplete) { Note "ALL DONE" } else { Note "pass ended with work left; the 30-minute watchdog re-enters" }
