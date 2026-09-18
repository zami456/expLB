<#
  Build the three label-set persona banks, then (optionally) run them through the 10 x 10.

    hf275_loose    published 275-role vectors, own100's loose labelling standard (120 roles)
    hf275_strict   published 275-role vectors, the paper's strict stance rule (46 roles)
    own100_strict  OUR 100-role vectors with the strict rule -- isolates the labelling rule
                   from the vector source, since hf100 ~= own100 on every measured number

  Scheduled Task or a short SSH call; each step is idempotent.
#>
param([switch]$SkipFetch)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"          # has huggingface_hub
$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"      # has torch for 6_axes
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "label_banks.status"
$log = Join-Path $logDir "label_banks.log"
"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Encoding utf8 $status

if (-not $SkipFetch -and -not (Test-Path "run-hf275\vectors_275\default.pt")) {
    & $pyAA "$D\31_fetch_hf_vectors.py" --out "run-hf275\vectors_275" 2>&1 |
        ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    "FETCH exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $status
}

$builds = @(
    @{ out = "run-hf275\axes_hf275_loose.pt";  vec = "run-hf275\vectors_275"; roles = "$D\hf275_role_groups_loose.json" },
    @{ out = "run-hf275\axes_hf275_strict.pt"; vec = "run-hf275\vectors_275"; roles = "$D\hf275_role_groups_strict.json" },
    @{ out = "run\axes_strict.pt";             vec = "run\vectors";           roles = "$D\own100_role_groups_strict.json" }
)
foreach ($b in $builds) {
    if (-not (Test-Path $b.vec)) { "SKIP $($b.out) (no $($b.vec))" | Out-File -Append -Encoding utf8 $status; continue }
    & $py "$D\6_axes.py" --vectors_dir $b.vec --output $b.out --roles_file $b.roles --report_layer 40 2>&1 |
        ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    "BUILT $($b.out) exit $LASTEXITCODE" | Out-File -Append -Encoding utf8 $status
}
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
