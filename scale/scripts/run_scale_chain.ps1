<#
  Stage 1 of the scale ladder: OUR pipeline, unchanged, at the paper's own model sizes.

  WHY. The paper reports +0.10 / +0.12 mean off-diagonal AUROC for persona-PC probes at 3B and
  8B. We get nothing positive at 70B. Two explanations are currently indistinguishable:

      (a) SCALE      -- the method works small and stops working large
      (b) RECONSTRUCTION -- our personas, questions, corpus or probes differ from theirs

  The paper released no code and no data, so everything on our side is a reconstruction. Running
  the IDENTICAL pipeline at 3B and 8B separates (a) from (b): reproduce their +0.10 at 3B and the
  70B result is a scale finding; fail to and it is a replication finding. Either way it is the
  biggest hole in the writeup, and docs/thesis/persona-coordinates-70b-results.md section 14.3
  already argued for it.

  ON-POLICY. Each model generates its OWN persona rollouts, which is the paper's design (section
  4.1) and what this chain does. The evaluation corpus is a separate problem -- the 15,278 rows
  we have were written by Llama-3.3-70B, so running 3B activations over them would be off-policy,
  the Q11 situation the project already concluded must never be the headline. Stage 2 regenerates
  the self-labelling datasets on-policy; this chain does not touch them.

  THE JUDGE IS HELD CONSTANT at Llama-3.1-70B across every rung. The judge is not the subject;
  varying it would confound judge quality with model scale.

  Models (both verified HOT on NDIF 2026-09-20):
      3b  meta-llama/Llama-3.2-3B-Instruct   28 layers, hidden 3072, paper's layer 14 = midpoint
      8b  meta-llama/Llama-3.1-8B-Instruct   32 layers, hidden 4096, paper's layer 16 = midpoint
  The paper's own "Llama 3-8B" is COLD and unnamed in the paper; 3.1-8B-Instruct is the
  defensible stand-in and section 14.1 flags the ambiguity.

  Bank: paper16 (26 personas) on the 20 shared questions -- the same inputs as
  run-paper16-shared at 70B, so the three rungs differ ONLY in model.

      run_scale_chain.ps1 -Tag 3b
      run_scale_chain.ps1 -Tag 8b
#>
param(
    [ValidateSet("3b", "8b")][string]$Tag = "3b",
    [switch]$Force,
    # the HF gate polls rather than failing; this bounds the wait
    [int]$WaitForAccessHours = 48
)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$MODEL = @{ "3b" = "meta-llama/Llama-3.2-3B-Instruct"
            "8b" = "meta-llama/Llama-3.1-8B-Instruct" }[$Tag]
$LAYERS = @{ "3b" = 28; "8b" = 32 }[$Tag]
$MID    = @{ "3b" = 14; "8b" = 16 }[$Tag]

$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$SRCDIR = Join-Path $repo "src\assistant_axis_ndif"
$out = Join-Path $repo "run-scale\$Tag"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir, $out | Out-Null
$status = Join-Path $logDir "scale_$Tag.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Run($stageLog, [string[]]$a) {
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir $stageLog)
    return $LASTEXITCODE
}

if (-not $Force -and (Test-Path $status) -and
    ((Get-Content $status -EA SilentlyContinue) -match '^DONE ').Count -gt 0) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " tag=$Tag model=$MODEL" |
    Out-File -Encoding utf8 $status

# NO WAIT GATE. The original version blocked on KD-Chain/KD-S23, which was wrong: those use
# meta-llama/Llama-3.3-70B-Instruct and this rung uses 3B or 8B. NDIF serves them as SEPARATE
# deployments, so the two jobs genuinely run in parallel and do not share a queue. The gate
# cost hours of wall clock for nothing.
#
# Only the JUDGE stage touches the 70B, and it is a small fraction of the chain.
#
# SECOND API KEY (optional). If NDIF_API_KEY_B is present in the pipeline .env, this rung uses
# it, so per-key rate limits cannot couple the two jobs. The value is never echoed; it is read
# from the file and set in this process tree only. python-dotenv's load_dotenv does NOT
# override an existing environment variable (override=False by default), so setting it here
# wins over the NDIF_API_KEY line in the same file.
$envFile = Join-Path $SRCDIR ".env"
if (Test-Path $envFile) {
    $b = (Get-Content $envFile | Where-Object { $_ -match '^\s*NDIF_API_KEY_B\s*=' } | Select-Object -First 1)
    if ($b) {
        $env:NDIF_API_KEY = ($b -split '=', 2)[1].Trim().Trim('"').Trim("'")
        Note "using NDIF_API_KEY_B for this rung (value not logged)"
    } else {
        Note "no NDIF_API_KEY_B found; using the primary key"
    }
}
Note "starting (no wait gate: 3B/8B are separate NDIF deployments from the 70B)"

$roles = Join-Path $SRCDIR "pc_role_groups.json"
$instr = Join-Path $SRCDIR "pc_instructions"
$qfile = Join-Path $SRCDIR "pc_questions_shared.jsonl"
$nQ = (Get-Content $qfile | Where-Object { $_.Trim() }).Count
$nI = (Get-ChildItem "$instr\*.json").Count
Note "inputs: $nI personas, $nQ questions (expect 26 / 20)"
if ($nI -lt 26 -or $nQ -ne 20) { Note "unexpected inputs -- stopping"; exit 1 }

# ---- HuggingFace gate --------------------------------------------------------------
# nnsight builds a local META model (no weights) to learn the architecture, and that needs
# config.json from HuggingFace. Every Llama repo is gated SEPARATELY, so a token with access
# to the 70B does not imply access to the 3B. The first attempt died 4 seconds in with
# "403 ... you are not in the authorized list" and 0 rollouts.
#
# Gating is per-ACCOUNT, not per-token: once the account is approved the existing token works
# with no config change. So poll rather than fail -- the rung then starts itself the moment
# Meta grants access, with no human in the loop.
$envFile = Join-Path $SRCDIR ".env"
$hf = ""
if (Test-Path $envFile) {
    $l = (Get-Content $envFile | Where-Object { $_ -match '^\s*HF_TOKEN\s*=' } | Select-Object -First 1)
    if ($l) { $hf = ($l -split '=', 2)[1].Trim().Trim('"').Trim("'") }
}
$cfgUrl = "https://huggingface.co/$MODEL/resolve/main/config.json"
$gateDeadline = (Get-Date).AddHours($WaitForAccessHours)
while ($true) {
    $ok = $false
    try {
        $null = Invoke-WebRequest -Uri $cfgUrl -Headers @{ Authorization = "Bearer $hf" } `
                    -TimeoutSec 30 -UseBasicParsing
        $ok = $true
    } catch { $ok = $false }
    if ($ok) { Note "HF access to $MODEL confirmed"; break }
    if ((Get-Date) -gt $gateDeadline) { Note "GAVE UP: no HF access to $MODEL after $WaitForAccessHours h"; exit 2 }
    Note "waiting for HF access to $MODEL (gated repo; request it as the token's account)"
    Start-Sleep -Seconds 600
}

Note "preflight"
Run "scale_${Tag}_preflight.log" @("$SRCDIR\0_preflight_ndif.py", "--warn-only") | Out-Null

# ---- generate: 26 x 5 x 20 = 2,600 rollouts, written BY THIS MODEL -------------------
Note "generating 26 x 5 x 20 = 2600 rollouts on $MODEL"
$code = Run "scale_${Tag}_generate.log" @("$SRCDIR\1_generate_ndif.py",
    "--model", $MODEL, "--roles_dir", $instr, "--roles_file", $roles,
    "--questions_file", $qfile, "--question_count", "20",
    # BATCH SIZE IS SET BY A PER-PROCESS MEMORY CAP, NOT BY GPU SIZE. Measured 2026-09-20 on
    # the 3B deployment: GPU 3 had 44.42 GiB total and 36.73 GiB FREE, yet the process was
    # capped -- "7.37 GiB allowed" -- and the 3B weights alone occupy ~7.2 GiB of that,
    # leaving 80-200 MiB of headroom. The allocation scales linearly with gen_batch:
    #     96 -> 140 MiB (fails)   24 -> 38 MiB (fails)   8 -> 14 MiB (84/100 rows, no retries)
    # gen_batch 96 was copied from the 70B chain, whose deployment has a far larger allowance
    # because an 80-layer model needs one. SMALL models get a SMALLER allowance, so the 70B
    # settings are exactly backwards here.
    "--output_dir", "$out\responses", "--gen_batch", "8", "--session_batches", "4",
    "--retries", "6")
$rows = 0
foreach ($f in (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue)) {
    $rows += (Get-Content $f.FullName | Where-Object { $_.Trim() }).Count
}
Note "generate exit ${code}: $((Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue).Count) files, $rows rollouts"
if ($rows -lt 2400) { Note "generation incomplete -- stopping (re-run resumes)"; exit 1 }

# ---- judge: SAME judge at every rung --------------------------------------------------
Note "judging with meta-llama/Llama-3.1-70B-Instruct (held constant across rungs)"
$code = Run "scale_${Tag}_judge.log" @("$SRCDIR\3_judge_ndif.py",
    "--responses_dir", "$out\responses", "--roles_dir", $instr, "--roles_file", $roles,
    "--output_dir", "$out\scores", "--judge_model", "meta-llama/Llama-3.1-70B-Instruct",
    "--judge_batch", "16", "--session_batches", "8")
Note "judge exit ${code}: $((Get-ChildItem "$out\scores\*.json" -EA SilentlyContinue).Count) score files"

# ---- vectors: ALL layers. 28 or 32 is cheaper than the 21-of-80 subset used at 70B, so
#      Pass A and Pass B collapse into one extraction at these sizes.
# EXTRACTION BATCH IS RUNG-SPECIFIC, and getting it wrong fails SILENTLY-ISH: on 2026-09-20
# the 3B ran batch_size 8 / session_batches 20 (copied from the 8B) and NDIF rejected every job
# 0.1s after dispatch. All 22 roles attempted produced "no vector produced" and the chain kept
# going, so the failure looked like slow progress rather than a dead run.
#
# The cause is F-21's per-PROCESS memory cap, not sequence length: the 3B rollouts are the same
# length as the 8B's (p50 1,371 vs 1,369 chars, max 2,909 vs 2,896), so nothing is over-long.
# What differs is headroom -- 3B weights leave only ~140 MiB, and an extraction holds hidden
# states for ALL 28 layers x batch x tokens x 3072, which dwarfs generation's footprint at the
# same batch. Measured: batch_size 2 / session_batches 4 completes a role in ~60s.
$actBatch = if ($MODEL -match "3\.2-3B") { "2" } else { "8" }
$actSess  = if ($MODEL -match "3\.2-3B") { "4" } else { "20" }
Note "extraction batching: batch_size=$actBatch session_batches=$actSess"

foreach ($v in @(@{ tag = "vectors_all"; keep = "0,1,2,3" }, @{ tag = "vectors_s23"; keep = "2,3" })) {
    Note "extracting $($v.tag) (all $LAYERS layers)"
    $code = Run "scale_${Tag}_$($v.tag).log" @("$SRCDIR\2_activations_ndif.py",
        "--model", $MODEL, "--responses_dir", "$out\responses", "--scores_dir", "$out\scores",
        "--output_dir", "$out\$($v.tag)", "--roles_file", $roles,
        "--keep_scores", $v.keep, "--min_count", "20", "--max_length", "8192",
        "--layers", "all", "--batch_size", "$actBatch", "--session_batches", "$actSess")
    Note "$($v.tag) exit ${code}: $((Get-ChildItem "$out\$($v.tag)\*.pt" -EA SilentlyContinue).Count) vectors"
}

# ---- axes, reported at the paper's own analysis layer for this size -------------------
# DF-48 fix (audit, 2026-09-23): this loop built BOTH vector sets but wrote every iteration to
# the same "$out\axes_$tag.pt" -- the loop variable never reached the output path -- so the
# vectors_all (unfiltered) bank was built and then silently overwritten by vectors_s23. The log
# still said "axes_vectors_all ... axes_vectors_s23", which is why it went unnoticed.
# Each arm now gets its own file. `axes_$tag.pt` is still written, from vectors_s23, because
# run_scale_crossgen.ps1:53 reads exactly that path -- so downstream behaviour is unchanged.
foreach ($vt in @("vectors_all", "vectors_s23")) {
    if ((Get-ChildItem "$out\$vt\*.pt" -EA SilentlyContinue).Count -lt 20) {
        Note "$vt too few vectors -- axes not built"; continue
    }
    $code = Run "scale_${Tag}_axes.log" @("$SRCDIR\6_axes.py", "--vectors_dir", "$out\$vt",
        "--output", "$out\axes_${tag}_$vt.pt", "--roles_file", $roles, "--report_layer", "$MID")
    Note "axes_${tag}_$vt exit $code (reported at layer $MID of $LAYERS)"
    if ($vt -eq "vectors_s23" -and $code -eq 0) {
        Copy-Item "$out\axes_${tag}_$vt.pt" "$out\axes_$tag.pt" -Force
        Note "axes_$tag.pt <- axes_${tag}_$vt.pt (the bank run_scale_crossgen.ps1 reads)"
    }
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_scale.ps1") -OnBusy Wait -Tag "$Tag/final"
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
