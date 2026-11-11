<#
  Alpaca control set for a model Liars' Bench never ran (Mistral-Small-24B-2501, owner 2026-10-08 "go for a"), then
  D5 + K and D9 + K. Stages, each resumable and each skipped once its output exists:
    1 generate   xmodel/alpaca_gen.py, 3 shards in parallel (at most 3 NDIF requests per model), until all 2,000
                 instructions have a reply (Liars' Bench protocol: see alpaca_gen.py)
    2 finalize   -> <Run>\code\results\subsets\alpaca.csv in Liars' Bench format (csv model name -CsvModel)
    3 spans      8_coords_ndif.py --prepare_only, then check_spans.py: every pooled span must decode to the reply
    4 read       alpaca_chain.ps1 (the read Qwen used): all layers, mean over the reply
    5 D + K      xmodel/d9k.py for D9 and D5 on every bank that has them (s23, all, s3)
  Status: <Run>\logs\alpaca_gen.status. Live logs: -LogDir (outside the synced folder). Ends with a Drive copy.
  Scheduled Task XModel-mistral2501-alpaca (register_alpaca_task.ps1 -Gen).
  2026-10-09 (owner: "queue the mistral 3.1 alpaca read after K"): -WaitFor <status file> holds the chain until that file's
  last line ends in DONE (the model's K chain), so the two never share the laptop link; a model with Liars' Bench Alpaca
  rows (its code\results\subsets\alpaca.csv present) skips stages 1-2; and stage 5 first builds D5-D9 for any bank that
  has its persona vectors but no D9 yet (d_contrasts.py, checked against the pipeline's axes).
  Scheduled Task XModel-mistral31-alpaca (register_alpaca_task.ps1 -Read mistral31).
#>
param([string]$Run = "$env:USERPROFILE\xmodel-runs\Mistral-Small-24B-Instruct-2501",
      [string]$Model = "mistralai/Mistral-Small-24B-Instruct-2501",
      [string]$CsvModel = "mistral-small-24b-instruct-2501",
      [string]$KKey = "mistral-small-24b-2501",
      [string]$LogDir = "$env:USERPROFILE\xmodel-runs\alpaca-logs-mistral2501",
      [string]$DriveBase = "tamara:expLB/xmodel_2026-10-06",
      [string]$Rclone = "~\AppData\Local\Microsoft\WinGet\Packages\Rclone.Rclone_Microsoft.Winget.Source_8wekyb3d8bbwe\rclone-v1.75.1-windows-amd64\rclone.exe",
      [int]$Attempts = 6, [string]$WaitFor = "",
      # 2026-10-09 (Llama-405B): read a layer subset in float32 -- 126 x 16384 at all layers would be 8 GB in fp16
      [string]$Layers = "all", [ValidateSet("float16", "float32")][string]$StoreDtype = "float16")
$ErrorActionPreference = "Continue"
$repo = Split-Path $PSScriptRoot -Parent
$py = Join-Path $repo ".venv-aa-win\Scripts\python.exe"
$pyA = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$code = Join-Path $Run "code"
$S = Join-Path $code "src\assistant_axis_ndif"
$gen = Join-Path $Run "eval\alpaca_gen"
$csvOut = Join-Path $code "results\subsets\alpaca.csv"
New-Item -ItemType Directory -Force -Path $LogDir, (Join-Path $Run "logs"), $gen | Out-Null
$status = Join-Path $Run "logs\alpaca_gen.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
if ((Test-Path $status) -and ((Get-Content $status -Tail 1) -match '  DONE')) { exit 0 }
$env:PYTHONIOENCODING = "utf8"
$env:NDIF_MIN_DOWNLOAD_KBPS = "40"   # the laptop link shared by 3 shards (xmodel-qwen-laptop memory, 03:21)
$env:NDIF_DOWNLOAD_GRACE = "20"
function Replies {
    $c = "import json,glob,sys; s=set()`nfor f in glob.glob(sys.argv[1]+'/replies.*.jsonl'):`n    s|={json.loads(l)['index'] for l in open(f,encoding='utf-8') if l.strip()}`nprint(len(s))"
    return [int](& $py -c $c $gen)
}
Note "START model=$Model csv=$CsvModel run=$Run waitfor=$WaitFor"
if ($WaitFor) {
    $said = $false
    while (-not ((Test-Path $WaitFor) -and ((Get-Content $WaitFor -Tail 1) -match '  DONE$'))) {
        if (-not $said) { Note "waiting for $WaitFor to end in DONE"; $said = $true }
        Start-Sleep -Seconds 120
    }
    Note "gate open: $WaitFor is DONE"
}

# ---- 1 generate -----------------------------------------------------------------------------------------
if (-not (Test-Path $csvOut)) {
    for ($t = 1; $t -le $Attempts; $t++) {
        $n = Replies
        if ($n -ge 2000) { break }
        Note "generate attempt $t (have $n / 2000)"
        $procs = foreach ($i in 0..2) {
            $a = @("`"$PSScriptRoot\alpaca_gen.py`"", "--code", "`"$code`"", "--out", "`"$gen`"", "--model", $Model,
                   "--csv_model", $CsvModel, "--shard", "$i/3")
            $p = Start-Process -FilePath $py -ArgumentList $a -WorkingDirectory $repo -NoNewWindow -PassThru `
                    -RedirectStandardOutput (Join-Path $LogDir "gen.$i.log") -RedirectStandardError (Join-Path $LogDir "gen.$i.err.log")
            $null = $p.Handle
            $p
        }
        $procs | ForEach-Object { $_.WaitForExit() }
        Note "generate exits: $(($procs | ForEach-Object { $_.ExitCode }) -join ',')"
    }
    $n = Replies
    if ($n -lt 2000) { Note "STOP generate $n / 2000 after $Attempts attempts (a re-run resumes)"; exit 1 }
    # ---- 2 finalize ---------------------------------------------------------------------------------------
    $out = & $py "$PSScriptRoot\alpaca_gen.py" --code $code --out $gen --model $Model --csv_model $CsvModel --finalize 2>&1
    Note "finalize exit ${LASTEXITCODE}: $(($out | Select-Object -Last 12) -join ' ')"
    if (-not (Test-Path $csvOut)) { Note "STOP finalize wrote no csv"; exit 1 }
}

# ---- 3 spans (tokenizer only) ---------------------------------------------------------------------------
$env:XMODEL_CSV_MODEL = $CsvModel
$items = Join-Path $Run "eval\alpaca\items.pt"
if (-not (Test-Path $items)) {
    $out = & $py "$S\8_coords_ndif.py" --model $Model --datasets alpaca --output_dir "$Run\eval" --layers all `
            --pooling mean --span response --prepare_only 2>&1
    Note "prepare_only exit ${LASTEXITCODE}: $(($out | Select-Object -Last 2) -join ' ')"
}
$out = & $py "$S\check_spans.py" --model $Model --eval_dir "$Run\eval" --datasets alpaca 2>&1
$spanExit = $LASTEXITCODE
Note "check_spans exit ${spanExit}: $(($out | Select-Object -Last 3) -join ' ')"
if ($spanExit -ne 0) { Note "STOP span check failed"; exit 1 }

# ---- 4 read (all layers) --------------------------------------------------------------------------------
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "alpaca_chain.ps1") -Run $Run -Model $Model `
    -CsvModel $CsvModel -LogDir $LogDir -DriveBase $DriveBase -Rclone $Rclone -Layers $Layers -StoreDtype $StoreDtype
$rs = Join-Path $Run "logs\alpaca.status"
if (-not ((Test-Path $rs) -and ((Get-Content $rs -Tail 1) -match '  DONE'))) { Note "STOP the read is not DONE (see $rs)"; exit 1 }
Note "read DONE"

# ---- 5 D5 + K, D9 + K -----------------------------------------------------------------------------------
foreach ($bank in "s23", "all", "s3") {   # the contrasts themselves, if the persona run left them unbuilt
    $vec = Join-Path $Run "persona\vectors_$bank"; $axes = Join-Path $Run "persona\axes_vectors_$bank.pt"
    if ((Test-Path (Join-Path $Run "d9\D9_$bank.pt")) -or -not (Test-Path $vec) -or -not (Test-Path $axes)) { continue }
    $out = & $pyA "$PSScriptRoot\d_contrasts.py" --only "D5,D6,D7,D8,D9" --vectors $vec --out $Run --tag $bank `
            --axes $axes --model $Model 2>&1
    Note "D5-D9 $bank exit ${LASTEXITCODE}: $(($out | Where-Object { $_ -match 'check:|refus|Error' }) -join ' | ')"
}
$k = Join-Path $Run "K\$KKey\first\K.npz"
foreach ($d in "D9", "D5") {
    foreach ($bank in "s23", "all", "s3") {
        $src = Join-Path $Run ("{0}\{1}_{2}.pt" -f $d.ToLower(), $d, $bank)
        $dst = Join-Path $Run ("{0}k\{1}K_{2}.pt" -f $d.ToLower(), $d, $bank)
        if (-not (Test-Path $src) -or (Test-Path $dst)) { continue }
        $out = & $pyA "$PSScriptRoot\d9k.py" --name $d --d $src --alpaca "$Run\eval\alpaca" --k $k `
                --out (Split-Path $dst -Parent) --tag $bank 2>&1
        Note "$d + K $bank exit ${LASTEXITCODE}: $(($out | Select-Object -Last 4) -join ' | ')"
    }
}
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "sync_drive.ps1") `
    -Folder (Split-Path $Run -Leaf) -RunBase (Split-Path $Run -Parent) -DriveBase $DriveBase -Rclone $Rclone `
    -Tag "alpaca_gen" -SkipLiveLogs
Note "drive copy exit $LASTEXITCODE"
Note "DONE"
