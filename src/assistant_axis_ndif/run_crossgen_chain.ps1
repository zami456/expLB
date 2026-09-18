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
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$status = Join-Path $repo "logs\crossgen_chain.status"
$full = Join-Path $D "run_crossgen_full.ps1"
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status

$ALL = "paper16_famcentred paper16_shared own100 hf100 hf275_loose hf275_strict own100_strict"
$steps = @(
    @{ tag = "B_newbanks"; a = @("-Pass", "B",      "-Layers", "all", "-Banks", "hf275_loose hf275_strict own100_strict", "-Seeds", "0 1 2 3 4") },
    @{ tag = "A";          a = @("-Pass", "A",      "-Layers", "all", "-Banks", $ALL,                                      "-Seeds", "0 1 2 3 4") },
    @{ tag = "B_fine";     a = @("-Pass", "B_fine", "-Layers", "all", "-Banks", "paper16_shared own100 hf275_strict",      "-Seeds", "") },
    @{ tag = "P";          a = @("-Pass", "P",      "-Layers", "all", "-Banks", "",                                        "-Seeds", "") }
)
foreach ($s in $steps) {
    "STEP $($s.tag) start " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $full @($s.a)
    "STEP $($s.tag) exit $LASTEXITCODE " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
