<#
  Random persona bank: the xmodel persona chain (xmodel/run_chain.ps1) on the random bank, for one model.
  One run folder per model:

      <RunBase>\<Model>\code      what runs: xmodel/code's pipeline + this bank in code\rpb (register_task.ps1 deploys it)
      <RunBase>\<Model>\persona   responses -> scores -> vectors_s23 / vectors_all / vectors_s3 -> axes
      <RunBase>\<Model>\logs      chain.status (one line per event) + per-process logs

  Differences from xmodel/run_chain.ps1, and nothing else:
    - the bank: code\rpb\rpb_role_groups.json, code\rpb\rpb_instructions, code\rpb\questions (36 roles / 30 questions);
    - persona side only (no Liars' Bench eval pooling: K and the Alpaca reads already exist per model);
    - a llama33 tag (the kd bank's own model; it never had an xmodel run folder);
    - laptop paths are the defaults (RunBase = random_persona_bank\results).
  Same sampling (0.7 / 0.9 / top_k 0 / rep 1.0, 512 tokens), same judge (Llama-3.1-70B), same filters (s23, all, s3,
  min 20), every layer. At most three NDIF requests at a time.

  Every stage is resumable per role, so a stage that ends incomplete is run again (up to $Attempts times). Gates read
  xmodel_status.py, never an exit code. Started by queue.ps1 from a Scheduled Task, never Start-Process over SSH.
#>
param(
    [ValidateSet("llama33", "qwen", "mistral2501", "mistral31", "gemma3")][string]$Tag = "llama33",
    [int]$Attempts = 6,
    [switch]$Force,
    [string]$RunBase = "",
    [string]$Py = "",
    [string]$DriveBase = "tamara:expLB/random_persona_bank_2026-10-10",
    [string]$Rclone = "",
    [string]$MinDownloadKbps = "40",
    [string]$DownloadGrace = "20"
)
$ErrorActionPreference = "Continue"

# tag -> (model, folder, short name, report layer = half depth). Same folders as xmodel's runs.
$cfg = @{
    "llama33"     = @{ Model = "meta-llama/Llama-3.3-70B-Instruct"; Folder = "Llama-3.3-70B-Instruct"; Short = "Llama"; ReportLayer = "40" }
    "qwen"        = @{ Model = "Qwen/Qwen2.5-72B-Instruct"; Folder = "Qwen2.5-72B-Instruct"; Short = "Qwen"; ReportLayer = "40" }
    "mistral2501" = @{ Model = "mistralai/Mistral-Small-24B-Instruct-2501"; Folder = "Mistral-Small-24B-Instruct-2501"; Short = "Mistral"; ReportLayer = "20" }
    "mistral31"   = @{ Model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"; Folder = "Mistral-Small-3.1-24B-Instruct-2503"; Short = "Mistral"; ReportLayer = "20" }
    "gemma3"      = @{ Model = "google/gemma-3-27b-it"; Folder = "gemma-3-27b-it"; Short = "Gemma"; ReportLayer = "31" }
}[$Tag]

$Here = Split-Path $PSCommandPath -Parent            # <RunBase>\<Model>\code once deployed
if (-not $RunBase) { $RunBase = Split-Path (Split-Path $Here -Parent) -Parent }
$Judge = "meta-llama/Llama-3.1-70B-Instruct"
$Root  = "$RunBase\$($cfg.Folder)"
$Code  = "$Root\code"
$S     = "$Code\src\assistant_axis_ndif"
$P     = "$Root\persona"
$Logs  = "$Root\logs"
$py    = $Py
if (-not $py) { $py = Join-Path (Split-Path (Split-Path $RunBase -Parent) -Parent) ".venv-aa-win\Scripts\python.exe" }
$status = "$Logs\chain.status"
New-Item -ItemType Directory -Force -Path $P, $Logs | Out-Null
Set-Location $Code

$env:PYTHONIOENCODING  = "utf8"
$env:XMODEL_SHORT_NAME = $cfg.Short        # default.json's "You are {model_name}."
$env:XMODEL_CSV_MODEL  = "none"            # no Liars' Bench loader is used; refuse rather than read Llama's rows
if ($MinDownloadKbps) { $env:NDIF_MIN_DOWNLOAD_KBPS = $MinDownloadKbps }
if ($DownloadGrace)   { $env:NDIF_DOWNLOAD_GRACE = $DownloadGrace }

$roles = "$Code\rpb\rpb_role_groups.json"
$instr = "$Code\rpb\rpb_instructions"
$qfile = "$Code\rpb\questions\kd_questions_shared.jsonl"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

function Status() {
    $j = & $py "$S\xmodel_status.py" --root $Root --json 2>$null
    try { return ($j | Select-Object -Last 1 | ConvertFrom-Json) } catch { return $null }
}

# Start one python per argument list, wait for all. Output goes to one log per process.
function RunParallel($stage, $argLists) {
    $procs = @()
    $i = 0
    foreach ($a in $argLists) {
        $out = "$Logs\$stage.$i.log"
        $err = "$Logs\$stage.$i.err.log"
        $quoted = $a | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
        $p = Start-Process -FilePath $py -ArgumentList $quoted -WorkingDirectory $Code `
                    -RedirectStandardOutput $out -RedirectStandardError $err -NoNewWindow -PassThru
        $null = $p.Handle   # PS 5.1 reports ExitCode only if the handle was taken before exit
        $procs += $p
        $i++
    }
    $procs | Wait-Process
    return ($procs | ForEach-Object { $_.ExitCode })
}

function Sync($tag) {
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$Code\sync_drive.ps1",
           "-Folder", $cfg.Folder, "-Tag", $tag, "-RunBase", $RunBase, "-DriveBase", $DriveBase)
    if ($Rclone) { $a += @("-Rclone", $Rclone) }
    & powershell.exe @a
}

if (-not $Force -and (Test-Path $status) -and ((Get-Content $status -Tail 1) -match '^DONE ')) { exit 0 }
Note "START tag=$Tag model=$($cfg.Model) judge=$Judge root=$Root py=$py floor=$($env:NDIF_MIN_DOWNLOAD_KBPS) bank=rpb"

# ---- inputs: the random bank, checked against its manifest (the judge prompts are renamed at deploy) -------------
$nInstr = (Get-ChildItem "$instr\*.json").Count
$nQ = (Get-Content $qfile | Where-Object { $_.Trim() }).Count
$nGroups = & $py -c "import json,sys; g=json.load(open(sys.argv[1]))['groups']; print(sum(len(v['roles']) for v in g.values()))" $roles
Note "inputs: $nInstr role files, $nQ questions, $nGroups roles in the groups file (expect 36 / 30 / 36)"
if ($nInstr -ne 36 -or $nQ -ne 30 -or "$nGroups".Trim() -ne "36") { Note "STOP unexpected inputs"; exit 1 }

& $py "$S\0_preflight_ndif.py" --require $cfg.Model $Judge --warn-only 2>&1 |
    Out-File -Append -Encoding utf8 "$Logs\preflight.log"

# ---- 1 generate: 36 x 5 x 30 = 5,400 rollouts, written by THIS model ------------------------
for ($t = 1; $t -le $Attempts; $t++) {
    $st = Status
    if ($st -and $st.persona.n_rollouts -ge 5400) { break }
    Note "generate attempt $t (have $($st.persona.n_rollouts) / 5400)"
    $lists = 0..2 | ForEach-Object {
        ,@("$S\1_generate_ndif.py", "--model", $cfg.Model, "--roles_dir", $instr,
           "--roles_file", $roles, "--shard", "$_/3", "--questions_file", $qfile,
           "--question_count", "30", "--output_dir", "$P\responses",
           "--temperature", "0.7", "--top_p", "0.9", "--top_k", "0", "--repetition_penalty", "1.0",
           "--max_new_tokens", "512", "--gen_batch", "32", "--session_batches", "4", "--retries", "6")
    }
    $codes = RunParallel "generate" $lists
    Note "generate exits: $($codes -join ',')"
}
$st = Status
Note "generate: $($st.persona.n_rollouts) rollouts in $(@($st.persona.responses.PSObject.Properties).Count) role files"
Sync "generate"
if (-not $st -or $st.persona.n_rollouts -lt 5400) { Note "STOP generation incomplete (re-run resumes)"; exit 1 }

# ---- 2 judge (Llama-3.1-70B) ----------------------------------------------------------------
for ($t = 1; $t -le $Attempts; $t++) {
    $st = Status
    if ($st -and $st.persona.n_scored -ge ($st.persona.n_rollouts - $st.persona.responses.default)) { break }
    Note "judge attempt $t (scored $($st.persona.n_scored))"
    $lists = 0..2 | ForEach-Object {
        ,@("$S\3_judge_ndif.py", "--responses_dir", "$P\responses", "--roles_dir", $instr,
           "--roles_file", $roles, "--shard", "$_/3", "--output_dir", "$P\scores",
           "--judge_model", $Judge, "--judge_batch", "16", "--session_batches", "8")
    }
    $codes = RunParallel "judge" $lists
    Note "judge exits: $($codes -join ',')"
    Sync "judge"
}
$st = Status
Note "judge: $($st.persona.n_scored) scored"
if (-not $st -or $st.persona.n_scored -lt ($st.persona.n_rollouts - $st.persona.responses.default)) {
    Note "STOP judging incomplete (re-run resumes)"; exit 1
}

# ---- 3 persona vectors, every layer: s23 (the thesis bank), all, s3 ---------------------------
# "done" = every ELIGIBLE role has its vector (>= 20 replies with a kept score; default always).
function Eligible($keep) {
    $code = "import json,glob,os,sys; k={int(x) for x in sys.argv[2].split(',')}; " +
            "n=sum(1 for f in glob.glob(os.path.join(sys.argv[1],'*.json')) " +
            "if sum(v in k for v in json.load(open(f,encoding='utf-8')).values())>=20); print(n+1)"
    return [int](& $py -c $code "$P\scores" $keep)
}
$short = $false
foreach ($v in @(@{ tag = "vectors_s23"; keep = "2,3" }, @{ tag = "vectors_all"; keep = "0,1,2,3" },
                 @{ tag = "vectors_s3"; keep = "3" })) {
    $want = Eligible $v.keep
    for ($t = 1; $t -le $Attempts; $t++) {
        $have = (Get-ChildItem "$P\$($v.tag)\*.pt" -EA SilentlyContinue).Count
        if ($have -ge $want) { break }
        Note "$($v.tag) attempt $t (have $have / $want eligible of 36)"
        $lists = 0..2 | ForEach-Object {
            ,@("$S\2_activations_ndif.py", "--model", $cfg.Model, "--responses_dir", "$P\responses",
               "--scores_dir", "$P\scores", "--output_dir", "$P\$($v.tag)", "--roles_file", $roles,
               "--shard", "$_/3", "--keep_scores", $v.keep, "--min_count", "20",
               "--max_length", "8192", "--layers", "all", "--batch_size", "8", "--session_batches", "20")
        }
        $codes = RunParallel $v.tag $lists
        Note "$($v.tag) exits: $($codes -join ',')"
    }
    $have = (Get-ChildItem "$P\$($v.tag)\*.pt" -EA SilentlyContinue).Count
    Note "$($v.tag): $have vectors ($want eligible of 36; min_count 20 on scores $($v.keep))"
    if ($have -lt $want) { $short = $true }
    Sync $v.tag
}
if ($short) { Note "STOP a bank is missing eligible vectors (re-run resumes)"; exit 1 }

# ---- 4 axes (assistant axis + all-roles PCA; the bank has no behaviour groups) ----------------
foreach ($bank in @("vectors_s23", "vectors_all", "vectors_s3")) {
    $nb = (Get-ChildItem "$P\$bank\*.pt" -EA SilentlyContinue).Count
    if ($nb -lt 30) { Note "$bank has $nb vectors (< 30) -- axes not built"; continue }
    & $py "$S\6_axes.py" --vectors_dir "$P\$bank" --output "$P\axes_$bank.pt" --roles_file $roles `
        --report_layer $cfg.ReportLayer 2>&1 | Out-File -Append -Encoding utf8 "$Logs\axes.log"
    $ax = Test-Path "$P\axes_$bank.pt"
    Note "axes_$bank exit $LASTEXITCODE, file written: $ax"
}

& $py "$S\xmodel_status.py" --root $Root 2>&1 | Out-File -Encoding utf8 "$Logs\final_status.txt"
Sync "final"
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
