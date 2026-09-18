<#
  Three NDIF extractions that close the gaps in the current stores (phase-2 plan, step 0).

    P0a  200 stratified sycophancy_mt rows into passA (all 80 layers) and passC (per-token at
         20/22/28/40/44/48). Same --sample 200 --seed 42 as those passes, so both stores hold
         the SAME rows and every arm becomes a 10 x 10 instead of 9 x 9.
    P0b  prompt-side activations (mask = [0, a0) instead of the assistant turn) at layers
         20/40/44 for the whole corpus -> passP. The construct-validity control: if a probe on
         prompt activations transfers as well as one on response activations, the matrix is
         largely detecting the instructed condition rather than the response.
    P0c  the nine layers missing around the peak -- 37,38,39,41,42,43,45,46,47 -- plus 44 again
         as an overlap control, for the whole corpus -> passB_fine. Resolves the peak to +/-1
         layer instead of +/-4. Stored separately because load_all_layers takes its layer list
         from the first chunk it reads.

  Every stage is resumable per chunk; re-running continues where it stopped. Scheduled Task
  only (register_extra_extractions_task.ps1), never over plain SSH.
#>
param(
    [string]$Jobs = "P0a P0b P0c"
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "extra_extractions.status"
$script = Join-Path $D "8_coords_ndif.py"

# the ten labelled datasets plus the control; ai_audit is excluded by decision (its contexts
# need more eager-attention memory than the shared GPU has free)
$CORPUS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
            "convincing-game", "insider-trading", "ai_liar", "sandbagging_v2", "roleplaying",
            "mask", "sycophancy_mt", "alpaca")

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
function Run($tag, [string[]]$a) {
    Note "START $tag"
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 (Join-Path $logDir "$tag.log")
    Note "EXIT $tag $LASTEXITCODE"
}

"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " jobs=$Jobs" | Out-File -Encoding utf8 $status

if ($Jobs -match "P0a") {
    Run "extract_P0a_passA" @($script, "--datasets", "sycophancy_mt", "--sample", "200", "--seed", "42",
                              "--layers", "all", "--output_dir", "run-coords\passA")
    Run "extract_P0a_passC" @($script, "--datasets", "sycophancy_mt", "--sample", "200", "--seed", "42",
                              "--layers", "20", "22", "28", "40", "44", "48", "--per_token",
                              "--output_dir", "run-coords\passC")
}
if ($Jobs -match "P0b") {
    Run "extract_P0b_passP" (@($script, "--datasets") + $CORPUS +
                             @("--span", "prompt", "--layers", "20", "40", "44",
                               "--output_dir", "run-coords\passP"))
}
if ($Jobs -match "P0c") {
    Run "extract_P0c_passB_fine" (@($script, "--datasets") + $CORPUS +
                                  @("--layers", "37", "38", "39", "41", "42", "43", "44", "45", "46", "47",
                                    "--output_dir", "run-coords\passB_fine"))
}

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (Test-Path $rc) {
    foreach ($p in @("passA", "passC", "passP", "passB_fine")) {
        $src = Join-Path $repo "run-coords\$p"
        if (Test-Path $src) {
            & $rc copy $src "tamara:expLB/coords/$p" --include "*.pt" --include "*.jsonl" `
                --checksum --transfers 4 --log-file (Join-Path $logDir "extra_extractions_rclone.log") --log-level INFO
            Note "RCLONE $p exit $LASTEXITCODE"
        }
    }
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
