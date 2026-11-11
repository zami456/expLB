<#
  xmodel: the thesis persona pipeline on another Liars' Bench model. One model per run folder.

      D:\user4fina\<Model>\code      what runs (a copy; nothing under ~\expLB is touched)
      D:\user4fina\<Model>\persona   responses -> scores -> vectors_s3 / vectors_all -> axes
      D:\user4fina\<Model>\eval      that model's own Liars' Bench rows, mean-pooled, every layer
      D:\user4fina\<Model>\logs      chain.status (one line per event) + per-process logs

  Order, and why. Each model gets at most THREE NDIF requests at a time (owner, 2026-09-29):
      1  generate       subject x3 shards
      2  judge          Llama-3.1-70B x3  ||  eval pooling  subject x3   (different deployments)
      3  vectors_s23, vectors_all, vectors_s3   subject x3 each (s23 = the thesis bank)
      4  axes           local
  The judge is held at Llama-3.1-70B, exactly as for the Llama kd bank, so the subject is the only
  thing that changes.

  Every stage is resumable per role / per chunk, so a stage that ends incomplete is simply run
  again (up to $Attempts times). Gates read xmodel_status.py, never an exit code.

  Scheduled Task only (register_task.ps1): Start-Process over SSH dies with the session.
#>
param(
    [ValidateSet("qwen", "mistral2501", "mistral31", "gemma3", "llama405", "llama33")][string]$Tag = "qwen",
    [int]$Attempts = 6,
    [switch]$Force,
    # 2026-10-06: machine-specific paths are parameters so one script serves the lab PC (defaults,
    # unchanged) and the laptop (register_task.ps1 -Machine laptop passes its own).
    [string]$RunBase = "D:\user4fina",
    [string]$Py = "~\expLB\.venv-aa\Scripts\python.exe",
    [string]$DriveBase = "tamara:expLB/user4fina_2026-09-29",
    [string]$Rclone = "",
    # persona vectors only: skip stage 2's Liars' Bench eval pooling (owner, 2026-10-06)
    [switch]$SkipEval,
    # laptop download floor (2026-10-07): three shards share the link at ~70-95 KB/s each, under the
    # lab-tuned 300 KB/s; empty = ndif_common's default
    [string]$MinDownloadKbps = "",
    [string]$DownloadGrace = "",
    # suffix for the vector banks and their axes (2026-10-07: "_v2" = rebuilt after the Qwen
    # header-newline span fix, written beside the superseded banks so nothing is overwritten)
    [string]$VecSuffix = "",
    # 2026-10-14 (F-130, owner: matched pair): "user_stay" = role line + one sentence in the USER turn
    # (1_generate_ndif.py --prompt_format); "system" = the original format, and then nothing is passed
    [ValidateSet("system", "user_stay")][string]$PromptFormat = "system",
    # skip the Drive copies (2026-10-14: the tamara: token expired; copy by hand after reconnecting)
    [switch]$NoSync,
    # seconds before an NDIF job with no result is abandoned and retried (ndif_common.session_deadline);
    # empty = no deadline. A job sat at RUNNING for 6 h on 2026-10-14.
    [string]$SessionTimeout = ""
)
$ErrorActionPreference = "Continue"

$cfg = @{
    "qwen" = @{ Model = "Qwen/Qwen2.5-72B-Instruct"; Folder = "Qwen2.5-72B-Instruct"
                Short = "Qwen"; Csv = "qwen-2.5-72b-instruct"; ReportLayer = "40"
                Datasets = @(@("instructed-deception"), @("harm-pressure-knowledge-report"),
                             @("harm-pressure-choice", "alpaca")) }
    # 2026-10-07 (owner): a second family. 2501 is HOT on NDIF as a text LanguageModel (40 x 5120) but has
    # no Liars' Bench rows of its own (Csv "none": every LB loader refuses); its sibling 3.1-2503 is the
    # benchmark's model, a vision-language checkpoint that is COLD on NDIF. ReportLayer 20 = half depth,
    # as 40 is for the 80-layer models.
    "mistral2501" = @{ Model = "mistralai/Mistral-Small-24B-Instruct-2501"; Folder = "Mistral-Small-24B-Instruct-2501"
                       Short = "Mistral"; Csv = "none"; ReportLayer = "20"; Datasets = @() }
    "mistral31"   = @{ Model = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"; Folder = "Mistral-Small-3.1-24B-Instruct-2503"
                       Short = "Mistral"; Csv = "mistral-small-3.1-24b-instruct"; ReportLayer = "20"
                       Datasets = @(@("instructed-deception"), @("harm-pressure-knowledge-report"),
                                    @("harm-pressure-choice", "insider-trading", "alpaca")) }
    # 2026-10-08 (owner): the fourth Liars' Bench model. A vision-language checkpoint (Gemma3ForConditionalGeneration,
    # 62 x 5376), COLD on NDIF, loaded like Mistral-3.1 (VisionLanguageModel, text only). Its template folds the system
    # prompt into the first user turn. Liars' Bench has no HP-KR or HP-C rows for Gemma. ReportLayer 31 = half depth.
    "gemma3"      = @{ Model = "google/gemma-3-27b-it"; Folder = "gemma-3-27b-it"
                       Short = "Gemma"; Csv = "gemma-3-27b-it"; ReportLayer = "31"
                       Datasets = @(@("instructed-deception"), @("convincing-game", "insider-trading"), @("alpaca")) }
    # 2026-10-09 (owner): the top of the Llama scale ladder (3B / 8B / 70B -> 405B). A text LlamaForCausalLM (126 x 16384),
    # the same chat template family as Llama-3.3 (static "26 Jul 2024"), no Liars' Bench rows (Csv "none"). ReportLayer 63.
    "llama405"    = @{ Model = "meta-llama/Llama-3.1-405B-Instruct"; Folder = "Llama-3.1-405B-Instruct"
                       Short = "Llama"; Csv = "none"; ReportLayer = "63"; Datasets = @() }
    # 2026-10-14 (owner: matched pair, F-130): the thesis model itself through this chain, for the user_stay re-run.
    # Persona vectors only (-SkipEval), so no Liars' Bench rows are read (Csv "none"). ReportLayer 40, as the thesis.
    "llama33"     = @{ Model = "meta-llama/Llama-3.3-70B-Instruct"; Folder = "Llama-3.3-70B-Instruct"
                       Short = "Llama"; Csv = "none"; ReportLayer = "40"; Datasets = @() }
}[$Tag]

$Judge = "meta-llama/Llama-3.1-70B-Instruct"
$Root  = "$RunBase\$($cfg.Folder)"
$Code  = "$Root\code"
$S     = "$Code\src\assistant_axis_ndif"
$P     = "$Root\persona"
$Ev    = "$Root\eval"
$Logs  = "$Root\logs"
$py    = $Py
$status = "$Logs\chain.status"
New-Item -ItemType Directory -Force -Path $P, $Ev, $Logs | Out-Null
Set-Location $Code

$env:PYTHONIOENCODING  = "utf8"
$env:XMODEL_SHORT_NAME = $cfg.Short
$env:XMODEL_CSV_MODEL  = $cfg.Csv
if ($MinDownloadKbps) { $env:NDIF_MIN_DOWNLOAD_KBPS = $MinDownloadKbps }
if ($DownloadGrace)   { $env:NDIF_DOWNLOAD_GRACE = $DownloadGrace }
if ($SessionTimeout)  { $env:NDIF_SESSION_TIMEOUT = $SessionTimeout }

$roles = "$Code\kd\personas\kd_role_groups.json"
$instr = "$Code\kd\personas\kd_instructions"
$qfile = "$Code\kd\questions\kd_questions_shared.jsonl"

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
    if ($NoSync) { Note "sync $tag skipped (-NoSync)"; return }
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "$Root\code\sync_drive.ps1",
           "-Folder", $cfg.Folder, "-Tag", $tag, "-RunBase", $RunBase, "-DriveBase", $DriveBase)
    if ($Rclone) { $a += @("-Rclone", $Rclone) }
    & powershell.exe @a
}

if (-not $Force -and (Test-Path $status) -and ((Get-Content $status -Tail 1) -match '^DONE ')) {
    # exit WITHOUT writing: a "SKIP" note used to become the last line, so the next watchdog trigger no longer saw DONE
    # and re-ran the whole chain (every stage skipped, axes rebuilt identically) -- seen on Mistral-3.1, 2026-10-09 01:56
    exit 0
}
Note "START tag=$Tag model=$($cfg.Model) judge=$Judge root=$Root py=$py skipEval=$SkipEval floor=$($env:NDIF_MIN_DOWNLOAD_KBPS) vecSuffix=$VecSuffix format=$PromptFormat noSync=$NoSync sessionTimeout=$($env:NDIF_SESSION_TIMEOUT)"

# ---- inputs: the kd bank, unchanged except the model name (checked, not assumed) ----------
$nInstr = (Get-ChildItem "$instr\*.json").Count
$nQ = (Get-Content $qfile | Where-Object { $_.Trim() }).Count
Note "inputs: $nInstr persona files, $nQ questions (expect 36 / 30)"
if ($nInstr -ne 36 -or $nQ -ne 30) { Note "STOP unexpected inputs"; exit 1 }

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
    if ($PromptFormat -ne "system") { $lists = $lists | ForEach-Object { ,($_ + @("--prompt_format", $PromptFormat)) } }
    $codes = RunParallel "generate" $lists
    Note "generate exits: $($codes -join ',')"
}
$st = Status
Note "generate: $($st.persona.n_rollouts) rollouts in $(@($st.persona.responses.PSObject.Properties).Count) role files"
Sync "generate"
if (-not $st -or $st.persona.n_rollouts -lt 5000) { Note "STOP generation incomplete (re-run resumes)"; exit 1 }

# ---- 2 judge (Llama-3.1-70B)  ||  eval pooling (subject) -------------------------------------
$judgeLists = 0..2 | ForEach-Object {
    ,@("$S\3_judge_ndif.py", "--responses_dir", "$P\responses", "--roles_dir", $instr,
       "--roles_file", $roles, "--shard", "$_/3", "--output_dir", "$P\scores",
       "--judge_model", $Judge, "--judge_batch", "16", "--session_batches", "8")
}
$evalLists = $cfg.Datasets | ForEach-Object {
    ,(@("$S\8_coords_ndif.py", "--model", $cfg.Model, "--datasets") + $_ +
      @("--output_dir", $Ev, "--layers", "all", "--pooling", "mean", "--span", "response"))
}
if ($SkipEval) { $evalLists = @(); Note "eval pooling SKIPPED (-SkipEval): persona vectors only" }
for ($t = 1; $t -le $Attempts; $t++) {
    $st = Status
    $judged = $st -and $st.persona.n_scored -ge ($st.persona.n_rollouts - $st.persona.responses.default)
    $pooled = $SkipEval -or ($st -and (@($st.eval.PSObject.Properties | Where-Object { $_.Value.items -eq $null -or $_.Value.done -lt $_.Value.items }).Count -eq 0) `
              -and (@($st.eval.PSObject.Properties).Count -ge 4))
    if ($judged -and $pooled) { break }
    $lists = @()
    if (-not $judged) { $lists += $judgeLists }
    if (-not $pooled) { $lists += $evalLists }
    Note "judge+eval attempt $t (scored $($st.persona.n_scored); judge=$(-not $judged) eval=$(-not $pooled))"
    $codes = RunParallel "judge_eval" $lists
    Note "judge+eval exits: $($codes -join ',')"
    Sync "judge_eval"
}
$st = Status
Note "judge: $($st.persona.n_scored) scored"
foreach ($k in $st.eval.PSObject.Properties) { Note "eval $($k.Name): $($k.Value.done)/$($k.Value.items) rows, row shape $($k.Value.row_shape -join 'x')" }

# ---- 3 persona vectors, every layer ---------------------------------------------------------
# 2026-10-07: three banks, as the Llama kd bank has them. vectors_s23 (scores 2 or 3) is the thesis bank
# (run_kd_s23.ps1; F-10: s3 is drift-selected; F-121's 36-persona bank is s23) -- the 09-29 plan omitted
# it. vectors_all and vectors_s3 are the robustness banks. On Qwen, 11 roles have < 20 score-3 replies,
# so "done" means every ELIGIBLE role has its vector (eligible: >= min_count replies with a kept score;
# default always), not a fixed 36.
function Eligible($keep) {
    $code = "import json,glob,os,sys; k={int(x) for x in sys.argv[2].split(',')}; " +
            "n=sum(1 for f in glob.glob(os.path.join(sys.argv[1],'*.json')) " +
            "if sum(v in k for v in json.load(open(f,encoding='utf-8')).values())>=20); print(n+1)"
    return [int](& $py -c $code "$P\scores" $keep)
}
foreach ($v in @(@{ tag = "vectors_s23$VecSuffix"; keep = "2,3" }, @{ tag = "vectors_all$VecSuffix"; keep = "0,1,2,3" },
                 @{ tag = "vectors_s3$VecSuffix"; keep = "3" })) {
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
    Sync $v.tag
}

# ---- 4 axes --------------------------------------------------------------------------------
# $bank, not $tag: PowerShell names are case-insensitive, so $tag IS the [ValidateSet("qwen")] parameter $Tag;
# assigning "vectors_s23" to it threw, the loop was skipped silently and the chain still wrote DONE
# (2026-10-07 04:20 -- no axes built). Same class of bug as run_cap_controls.ps1's $src/$SRC.
foreach ($bank in @("vectors_s23$VecSuffix", "vectors_all$VecSuffix", "vectors_s3$VecSuffix")) {
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
