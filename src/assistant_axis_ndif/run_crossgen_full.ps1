<#
  The full cross-generalization job: every pass, every bank, all layers, 6 splits.

      -Pass    B | B_fine | A | P        which activation store
      -Layers  "all" or an explicit list
      -Banks   space-separated bank names ("" = baseline/paper16 root run only)
      -Seeds   extra 80/20 splits beyond seed 42

  Each pass writes to its own folder so baselines and splits never cross stores:
      B       run-crossgen\            + run-crossgen\banks\<bank>
      B_fine  run-crossgen\passB_fine\ + ...\banks\<bank>
      A       run-crossgen\passA\      + ...\banks\<bank>
      P       run-crossgen\passP\      (prompt-side control, root only)

  passA holds 9 labelled datasets (no sycophancy_mt) unless the P0a extraction has landed, so the
  dataset list is passed explicitly per pass. Scheduled Task only.
#>
param(
    [string]$Pass = "B",
    [string]$Layers = "all",
    [string]$Banks = "paper16_famcentred paper16_shared own100 hf100",
    [string]$Seeds = "0 1 2 3 4"
)
$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$script = Join-Path $D "27_crossgen.py"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "crossgen_full_$Pass.status"
$log = Join-Path $logDir "crossgen_full_$Pass.log"

$TEN = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
         "convincing-game", "insider-trading", "ai_liar", "sandbagging_v2", "roleplaying",
         "mask", "sycophancy_mt")
$conf = @{
    "B"      = @{ run = "run-coords\passB";      root = "run-crossgen" }
    "B_fine" = @{ run = "run-coords\passB_fine"; root = "run-crossgen\passB_fine" }
    "A"      = @{ run = "run-coords\passA";      root = "run-crossgen\passA" }
    "P"      = @{ run = "run-coords\passP";      root = "run-crossgen\passP" }
    # passC is the per-token store (layers 20,22,28,40,44,48). Its chunks carry the same
    # ("pooled", n, K, H, 2) tensor as every other store alongside the per-token blocks, so
    # load_all_layers reads it unchanged -- this arm was simply never wired up, which is why
    # no pass-C cross-generalization numbers exist at any layer.
    "C"      = @{ run = "run-coords\passC";      root = "run-crossgen\passC" }
}[$Pass]
if (-not $conf) { throw "unknown -Pass $Pass" }

# a store without the sycophancy rows runs 9 x 9
$ds = $TEN
if (-not (Test-Path (Join-Path $repo ($conf.run + "\sycophancy_mt\rows.jsonl")))) {
    $ds = $TEN | Where-Object { $_ -ne "sycophancy_mt" }
}

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

# Push what just finished, rather than holding every bank hostage to the end of the step.
# OnBusy Skip for the intermediate calls: an upload must never delay the next bank, and
# whatever a skipped call would have carried is picked up by the next sync or by the final
# whole-tree one. See sync_crossgen.ps1 for the lock and the per-object cost of Drive.
$syncer = Join-Path $D "sync_crossgen.ps1"
$rel = ($conf.root -replace '^run-crossgen\\?', '')
function Sync([string]$sub, [string]$tag, [string]$onBusy) {
    if (-not (Test-Path $syncer)) { Note "SYNC $tag skipped (no sync_crossgen.ps1)"; return }
    # -SubPath is omitted rather than passed empty: an empty argument to a native command
    # can be dropped, and then -SubPath would swallow the NEXT switch and sync a path that
    # does not exist -- a silent no-upload, which is the failure this whole change is about.
    $a = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $syncer,
           "-Tag", "$Pass/$tag", "-OnBusy", $onBusy)
    if ($sub) { $a += @("-SubPath", $sub) }
    & powershell.exe @a
    Note "SYNC $tag exit $LASTEXITCODE"
}

function Crossgen([string[]]$extra, [string]$tag) {
    $a = @($script, "--run_dir", $conf.run, "--datasets") + $ds + $extra
    if ($Layers.Trim() -ne "all") { $a += @("--layers") + ($Layers.Trim() -split "\s+") }
    if ($Seeds.Trim()) { $a += @("--extra_seeds") + ($Seeds.Trim() -split "\s+") }
    Note "START $tag"
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    Note "EXIT $tag $LASTEXITCODE"
}

"START " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") + " pass=$Pass layers=$Layers banks=$Banks seeds=$Seeds" |
    Out-File -Encoding utf8 $status
Note ("datasets: " + ($ds -join ", "))

# 1. baseline + paper16 (all features, the row bootstrap, and the shared splits)
Crossgen @("--stage", "all", "--out_root", $conf.root, "--bank_name", "paper16",
           "--bank", "run-paper16\axes_vectors_s3_fp32.pt") "root"
Sync $rel "root" "Skip"

# 2. the other persona banks: persona features only, baselines merged from the root at summarize
$bankFile = @{
    "paper16_famcentred" = "run-paper16\axes_vectors_s3_fp32_famcentred.pt"
    "paper16_shared"     = "run-paper16-shared\axes_vectors_s3.pt"
    "own100"             = "run\axes.pt"
    "own100_strict"      = "run\axes_strict.pt"
    "hf100"              = "run-hf275\axes_hf100.pt"
    "hf275_loose"        = "run-hf275\axes_hf275_loose.pt"
    "hf275_strict"       = "run-hf275\axes_hf275_strict.pt"
}
foreach ($b in ($Banks.Trim() -split "\s+" | Where-Object { $_ })) {
    $bf = $bankFile[$b]
    if (-not $bf -or -not (Test-Path $bf)) { Note "SKIP $b (missing $bf)"; continue }
    Crossgen @("--stage", "all", "--out_root", ($conf.root + "\banks\" + $b), "--bank_name", $b,
               "--bank", $bf, "--baseline_root", $conf.root,
               "--features", "pc:deception", "axes", "axes_resid", "pc:unified", "pc:unified@2",
               "--extra_features", "axes", "pc:deception", "pc:unified") $b
    Sync $(if ($rel) { "$rel\banks\$b" } else { "banks\$b" }) $b "Skip"
}

# the whole tree, including the root summaries the per-bank syncs do not cover. This one
# waits for the lock: it is the last chance for this step's output to leave the lab PC.
Sync "" "final" "Wait"
"DONE " + (Get-Date -Format "yyyy-MM-dd HH:mm:ss") | Out-File -Append -Encoding utf8 $status
