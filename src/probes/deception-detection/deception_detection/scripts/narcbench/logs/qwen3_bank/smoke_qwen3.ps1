# GB1 smoke for the Qwen3-32B persona bank (narcbench, 2026-10-23): one deception role (pc_handler), instruction 0,
# the 30 shared questions; generate -> judge -> per-question store at layers 26-30, all in a smoke folder that is
# never merged into the bank. The same scripts, arguments and environment as run_chain.ps1 -Tag qwen3.
#
#   launched detached:  Invoke-CimMethod Win32_Process Create (from the PowerShell tool, never the Bash tool)
$ErrorActionPreference = "Continue"
$Root = "E:\narcbench\persona\Qwen3-32B"
$Code = "$Root\code"
$S    = "$Code\src\assistant_axis_ndif"
$Sm   = "$Root\smoke"
$py   = "E:\aletheia\venv08\Scripts\python.exe"
$status = "$Sm\smoke.status"
New-Item -ItemType Directory -Force -Path "$Sm\logs" | Out-Null
Set-Location $Code

$env:PYTHONIOENCODING = "utf8"
$env:XMODEL_SHORT_NAME = "Qwen"
$env:XMODEL_ENABLE_THINKING = "0"
if (-not $env:HF_HOME) { $env:HF_HOME = "E:\hf_cache" }
# the keys come from the project's .env (never printed), as run_chain.ps1 loads them
foreach ($k in @("NDIF_API_KEY", "HF_TOKEN")) {
    if (-not (Get-Item "env:$k" -EA SilentlyContinue)) {
        $line = Get-Content "D:\expLB\liars-bench\.env" -EA SilentlyContinue | Where-Object { $_ -match "^\s*$k\s*=" } | Select-Object -First 1
        if ($line) { Set-Item -Path "env:$k" -Value (($line -split "=", 2)[1].Trim().Trim('"').Trim("'")) }
    }
}

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

function Run($name, $argList) {
    Note "start $name"
    $quoted = $argList | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }
    $p = Start-Process -FilePath $py -ArgumentList $quoted -WorkingDirectory $Code `
            -RedirectStandardOutput "$Sm\logs\$name.log" -RedirectStandardError "$Sm\logs\$name.err.log" -NoNewWindow -PassThru
    $null = $p.Handle
    $p | Wait-Process
    Note "$name exit $($p.ExitCode)"
    return $p.ExitCode
}

$instr = "$Code\kd\personas\kd_instructions"
$qfile = "$Code\kd\questions\kd_questions_shared.jsonl"
Note "START smoke model=Qwen/Qwen3-32B role=pc_handler prompt=0 questions=30"
$g = Run "generate" @("$S\1_generate_ndif.py", "--model", "Qwen/Qwen3-32B", "--roles_dir", $instr, "--roles", "pc_handler",
        "--questions_file", $qfile, "--question_count", "30", "--prompt_indices", "0", "--output_dir", "$Sm\responses",
        "--temperature", "0.7", "--top_p", "0.9", "--top_k", "0", "--repetition_penalty", "1.0",
        "--max_new_tokens", "512", "--gen_batch", "16", "--session_batches", "2", "--retries", "6")
if ($g -ne 0) { Note "STOP generate failed"; exit 1 }
$j = Run "judge" @("$S\3_judge_ndif.py", "--responses_dir", "$Sm\responses", "--roles_dir", $instr, "--roles", "pc_handler",
        "--output_dir", "$Sm\scores", "--judge_model", "meta-llama/Llama-3.1-70B-Instruct", "--judge_batch", "8",
        "--session_batches", "4", "--retries", "6")
if ($j -ne 0) { Note "STOP judge failed"; exit 1 }
$v = Run "store" @("$S\2_activations_ndif.py", "--model", "Qwen/Qwen3-32B", "--responses_dir", "$Sm\responses",
        "--scores_dir", "$Sm\scores", "--output_dir", "$Sm\vectors_pq_s3", "--roles", "pc_handler", "--keep_scores", "3",
        "--min_count", "1", "--per_question", "--on_failure", "refuse", "--max_length", "8192", "--batch_size", "8",
        "--session_batches", "8", "--token_budget", "4000", "--retries", "6", "--layers", "26", "27", "28", "29", "30")
if ($v -ne 0) { Note "STOP store failed"; exit 1 }
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
