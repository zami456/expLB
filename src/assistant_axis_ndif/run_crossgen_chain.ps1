<#
  Runs the remaining cross-generalization arms back to back, unattended:

    1. pass B   the three label-set banks (hf275 loose/strict, own100 strict) -- the earlier
                Crossgen-Full-B run already did the root and the first four banks
    2. pass A   all 80 layers on the 2,200-row subsample, every bank (now 10 x 10: the
                sycophancy rows landed in P0a)
    3. pass B_fine  layers 37-47 on the full corpus, seed 42 only -- locates the peak to +/-1
    4. pass P   the prompt-side control, root only: if a probe on PROMPT activations transfers
                as well as one on response activations, the matrix is detecting the instructed
                condition rather than the response

  Each step is the same run_crossgen_full.ps1, so each is resumable and skips finished files.
#>
param(
    [switch]$Force,
    # space-separated step tags to run, e.g. "B_fine P". Default: all four, in order.
    [string]$Only = ""
)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$status = Join-Path $repo "logs\crossgen_chain.status"
$full = Join-Path $D "run_crossgen_full.ps1"

# The task carries an AtStartup trigger so an unprompted reboot resumes the chain instead of
# stranding it (see register_crossgen_chain_task.ps1). That means this script also fires on
# boots long after the work is finished, so a completed run must be cheap to re-enter: if the
# status file already says DONE, leave immediately rather than re-verifying 123k files
# against Drive. Delete the status file, or pass -Force, to make it run again.
if (-not $Force -and (Test-Path $status) -and
    ((Get-Content $status -Tail 1) -match '^DONE ')) {
    "SKIP " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " already DONE; -Force to re-run" |
        Out-File -Append -Encoding utf8 $status
    exit 0
}

"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status

$ALL = "paper16_famcentred paper16_shared own100 hf100 hf275_loose hf275_strict own100_strict"

# An EMPTY STRING ARGUMENT IS DROPPED by `powershell.exe -File`. On 2026-09-19 that silently
# killed the last two steps: `-Seeds ""` left -Seeds dangling and B_fine died with "Missing an
# argument for parameter 'Seeds'" in the same second it started, writing no log; `-Banks ""
# -Seeds ""` did the same to P. Both surfaced only as `exit 1`, so the chain reported DONE
# having run half its work. run_crossgen_full.ps1 already carries a comment about exactly this
# hazard for -SubPath -- the fix was made there and not here.
#
# The switch cannot simply be omitted: -Banks defaults to four banks, and for P "" means ROOT
# ONLY. So pass a single SPACE. It survives argument passing, and run_crossgen_full.ps1 splits
# with `-split "\s+" | Where-Object { $_ }`, which yields an empty list -- the intended meaning.
$steps = @(
    @{ tag = "B_newbanks"; pass = "B";      banks = "hf275_loose hf275_strict own100_strict"; seeds = "0 1 2 3 4" },
    @{ tag = "A";          pass = "A";      banks = $ALL;                                     seeds = "0 1 2 3 4" },
    @{ tag = "B_fine";     pass = "B_fine"; banks = "paper16_shared own100 hf275_strict";     seeds = "" },
    @{ tag = "P";          pass = "P";      banks = "";                                       seeds = "" }
)
if ($Only) {
    $want = $Only.Trim() -split "\s+"
    $steps = $steps | Where-Object { $want -contains $_.tag }
    if (-not $steps) { throw "-Only '$Only' matched no step" }
}
foreach ($s in $steps) {
    $banks = if ($s.banks) { $s.banks } else { " " }
    $seeds = if ($s.seeds) { $s.seeds } else { " " }
    $a = @("-Pass", $s.pass, "-Layers", "all", "-Banks", $banks, "-Seeds", $seeds)
    "STEP $($s.tag) start " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $full @a
    "STEP $($s.tag) exit $LASTEXITCODE " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
