<#
  10 x 10 cross-generalization for the OTHER persona banks, on the baseline run's splits.

  Each bank writes only its persona features (deception PCs, contrast axes, unified PCs) to
  run-crossgen/banks/<name>/; raw, random, dataset PCA and length come from run-crossgen/ itself,
  so every comparison is cell by cell on identical rows.

      -Layers "40"          which layers (default 40; "all" = every stored layer)
      -Banks  "paper16_famcentred own100 hf100"

  Scheduled Task only (register_crossgen_banks_task.ps1), never over plain SSH.
#>
param(
    [string]$Layers = "40",
    [string]$Banks = "paper16_famcentred own100 hf100"
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$log = Join-Path $repo "logs\crossgen_banks.log"
$status = Join-Path $repo "logs\crossgen_banks.status"
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " layers=$Layers banks=$Banks" | Out-File -Encoding utf8 $status

$bankFile = @{
    "paper16_famcentred" = "run-paper16\axes_vectors_s3_fp32_famcentred.pt"
    "own100"             = "run\axes.pt"
    "hf100"              = "run-hf275\axes_hf100.pt"
    "hf275"              = "run-hf275\axes_hf275.pt"
    "paper16_shared"     = "run-paper16-shared\axes_vectors_s3.pt"
}

# the question-family-centred paper16 bank is derived from the fp32 vectors, so build it here
if ($Banks -match "paper16_famcentred" -and -not (Test-Path $bankFile["paper16_famcentred"])) {
    & $py "$D\29_bank_variants.py" --bank "run-paper16\axes_vectors_s3_fp32.pt" `
        --vectors_dir "run-paper16\vectors_s3_fp32" --out $bankFile["paper16_famcentred"] 2>&1 |
        ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    "FAMCENTRED_BUILT $LASTEXITCODE" | Out-File -Append -Encoding utf8 $status
}

foreach ($b in ($Banks.Trim() -split "\s+")) {
    $bf = $bankFile[$b]
    if (-not $bf -or -not (Test-Path $bf)) { "SKIP $b (bank file missing: $bf)" | Out-File -Append -Encoding utf8 $status; continue }
    $pyArgs = @("$D\27_crossgen.py", "--stage", "all", "--bank", $bf, "--bank_name", $b,
                "--out_root", "run-crossgen\banks\$b",
                "--features", "pc:deception", "axes", "axes_resid", "pc:unified", "pc:unified@2")
    if ($Layers.Trim() -ne "all") { $pyArgs += @("--layers") + ($Layers.Trim() -split "\s+") }
    "RUN $b " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
    & $py @pyArgs 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    "EXIT $b $LASTEXITCODE " + (Get-Date -Format "HH:mm:ss") | Out-File -Append -Encoding utf8 $status
}

$rc = Join-Path $repo "tools\rclone.exe"
if (Test-Path $rc) {
    & $rc copy "run-crossgen" "tamara:expLB/crossgen" --checksum --transfers 4 `
        --log-file (Join-Path $repo "logs\crossgen_banks_rclone.log") --log-level INFO
    "RCLONE_EXIT $LASTEXITCODE" | Out-File -Append -Encoding utf8 $status
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
