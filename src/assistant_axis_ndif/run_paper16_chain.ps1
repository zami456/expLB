<#
  Run the whole paper16 bank unattended: generate -> judge -> vectors -> axes -> gate.

  Exists because nothing supervises this overnight. Claude only runs while a conversation
  turn is active, so this is a plain detached process. Every stage is resumable -- stage 1
  per (role, prompt_index, question_index), stage 3 per scored response, stage 2 per role
  vector file -- so a sleep, a crash or a kill costs only the session in flight. Re-running
  this script continues from wherever it stopped.

  It produces TWO vector banks from one generation run, which is the point:

    vectors_all  every rollout kept. This is paper-faithful: Mahadik & Skapars apply no
                 judge filter at all.
    vectors_s3   only judge score 3 kept, matching how own100 was built. Without this the
                 paper16-vs-own100 comparison would differ in persona set AND in
                 filtering, reintroducing the confound the whole exercise exists to remove.

  The last step prints the number the night is actually for: cos(deception contrast,
  assistant axis). own100 measured -0.729/-0.710/-0.660 at layers 20/22/40 and the
  published HF vectors -0.686/-0.662/-0.610. If paper16 lands far below that, the
  contamination was the persona set. If it lands near it, it is a property of the model.

  Launch detached:
      Start-Process powershell -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass',
          '-File','src\assistant_axis_ndif\run_paper16_chain.ps1' -WindowStyle Hidden
#>
$ErrorActionPreference = "Continue"
Set-Location "C:\Users\SURAIYA\Documents\STUDY\thesis\expLB"

$py   = ".\.venv-aa-win\Scripts\python.exe"
$D    = "src\assistant_axis_ndif"
$out  = "run-paper16"
$log  = "logs\paper16_chain.log"
New-Item -ItemType Directory -Force -Path logs | Out-Null

function Log($m) {
    $line = "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m"
    $line | Add-Content $log
    Write-Host $line
}

function Wait-ForGeneration {
    while ($true) {
        # Poll rather than Wait-Process: the venv python is a trampoline that re-execs, so
        # the pid worth waiting on is not the one that survives.
        $p = Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
             Where-Object { $_.CommandLine -match '1_generate_ndif' }
        if (-not $p) { return }
        Start-Sleep -Seconds 30
    }
}

Log "=== chain started (pid $PID) ==="

# ---- stage 1b: sycophancy generation -------------------------------------------------
Wait-ForGeneration
Log "deception half done; starting sycophancy generation"
& $py "$D\1_generate_ndif.py" `
    --model meta-llama/Llama-3.3-70B-Instruct `
    --roles_dir "$D\pc_instructions" --roles_file "$D\pc_role_groups.json" `
    --groups sycophancy anti_sycophancy `
    --questions_file "$D\pc_questions_sycophancy.jsonl" --question_count 12 `
    --output_dir "$out\responses" --gen_batch 96 --session_batches 8 `
    *>> "logs\paper16_sycophancy.log"
$n = (Get-ChildItem "$out\responses\*.jsonl" -EA SilentlyContinue).Count
Log "generation complete: $n role files (expect 26)"
if ($n -lt 20) { Log "TOO FEW ROLE FILES -- stopping before judging"; exit 1 }

# ---- stage 3: judge ------------------------------------------------------------------
# Llama-3.1-70B-Instruct is pinned and was HOT through the outage, unlike the subject model.
#
# judge_batch 16, not the default 48. Measured against the longest paper16 prompts, a call
# costs len(batch) * longest-in-batch with left padding: batch 16 = 11,680 padded tokens
# and completes; batch 24 = 17,520 and NDIF rejects it 0.3s after dispatch. The default 48
# sends 35,040 and failed every single job, which is why the first overnight run produced
# 0 scores. Same class of bug as row-count batching in extraction.
Log "judging with meta-llama/Llama-3.1-70B-Instruct (judge_batch 16 -- 48 is over the limit)"
& $py "$D\3_judge_ndif.py" `
    --responses_dir "$out\responses" --roles_dir "$D\pc_instructions" `
    --roles_file "$D\pc_role_groups.json" --output_dir "$out\scores" `
    --judge_model meta-llama/Llama-3.1-70B-Instruct `
    --judge_batch 16 --session_batches 8 `
    *>> "logs\paper16_judge.log"
Log "scores on disk: $((Get-ChildItem "$out\scores\*.json" -EA SilentlyContinue).Count)"

# ---- stage 2: persona vectors, twice -------------------------------------------------
# min_count 20, not the default 50: the paper's elicitation budget is 5 variants x 10
# questions = 50 rollouts per persona, so score-3 filtering leaves ~48 and every role
# would be silently skipped at 50. max_length 8192 because head-truncation at 2048 drops
# long-context rows entirely (see build_items).
foreach ($v in @(
    @{ tag = "vectors_all"; keep = "0,1,2,3"; note = "unfiltered -- paper-faithful" },
    @{ tag = "vectors_s3";  keep = "3";       note = "score 3 only -- matches own100" }
)) {
    Log "extracting $($v.tag) ($($v.note))"
    & $py "$D\2_activations_ndif.py" `
        --model meta-llama/Llama-3.3-70B-Instruct `
        --responses_dir "$out\responses" --scores_dir "$out\scores" `
        --output_dir "$out\$($v.tag)" `
        --roles_file "$D\pc_role_groups.json" `
        --keep_scores $v.keep --min_count 20 --max_length 8192 `
        --layers all --batch_size 8 --session_batches 20 `
        *>> "logs\paper16_$($v.tag).log"
    Log "$($v.tag): $((Get-ChildItem "$out\$($v.tag)\*.pt" -EA SilentlyContinue).Count) vectors"
}

# ---- stage 6: axes -------------------------------------------------------------------
foreach ($tag in @("vectors_all", "vectors_s3")) {
    if ((Get-ChildItem "$out\$tag\*.pt" -EA SilentlyContinue).Count -lt 6) {
        Log "$tag has too few vectors for an axis; skipping"
        continue
    }
    Log "building axes for $tag"
    & $py "$D\6_axes.py" --vectors_dir "$out\$tag" --output "$out\axes_$tag.pt" `
        --roles_file "$D\pc_role_groups.json" --report_layer 40 `
        *>> "logs\paper16_axes.log"
}

# ---- the gate ------------------------------------------------------------------------
Log "=== COSINE GATE: is paper16 less assistant-contaminated than own100? ==="
& $py "$D\pc_gate.py" *>> $log

Log "=== chain finished ==="
