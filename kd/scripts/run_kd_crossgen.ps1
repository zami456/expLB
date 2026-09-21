<#
  11-dataset cross-generalization on the Pass B store, in the order the owner asked for:
  layers 40 and 44 first, then 20, then the remaining 18.

  The 11th dataset is `harm-pressure-kr-matched` -- the 258 HP-KR rows whose responses ALL
  deny knowledge, so wording is held constant and only the underlying knowledge state
  differs. It needs no new activations: 9_persona_probes.load_dataset keys rows by row_id and
  skips any chunk row absent from rows.jsonl, so a directory holding a 258-line rows.jsonl
  plus hardlinks to the same chunks loads exactly that slice. Built by make_view.py.

  No code change to 27_crossgen.py: --datasets rebinds DATASETS and BLOCK9 (line 800).
  Output goes to run-crossgen\kd11 -- a separate baseline root, so the existing 10-dataset
  results are never mixed with an 11-dataset design. The guard requires "run-crossgen" as an
  exact path component, which is why it is not called run-crossgen11.

  Why this order: every conclusion in the project is read at layer 40, and Q6 put the persona
  peak in the 36-48 window, so 40 and 44 answer the question; 20 is Apollo's layer and the
  one where HP-KR inverted worst; the rest is for the layer profile.

  CPU ONLY -- 27_crossgen.py contains no nnsight and runs in .venv-analysis. Nothing here
  touches NDIF.
#>
param([string]$Only = "")

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$script = Join-Path $repo "src\assistant_axis_ndif\27_crossgen.py"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "kd_crossgen.status"
$log = Join-Path $logDir "kd_crossgen.log"

$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

$steps = @(
    @{ tag = "peak";  layers = @("40", "44") },
    @{ tag = "l20";   layers = @("20") },
    @{ tag = "rest";  layers = @("0","4","8","12","16","22","24","28","32","36","48","52",
                                 "56","60","64","68","72","76") }
)
if ($Only) {
    $want = $Only.Trim() -split "\s+"
    $steps = $steps | Where-Object { $want -contains $_.tag }
    if (-not $steps) { throw "-Only '$Only' matched no step" }
}

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

Note "START 11-dataset crossgen on passB ($($DS.Count) datasets)"
foreach ($s in $steps) {
    Note "STEP $($s.tag) layers=$($s.layers -join ',') start"
    $a = @("$script", "--run_dir", "run-coords\passB", "--out_root", "run-crossgen\kd11",
           "--datasets") + $DS + @("--layers") + $s.layers + @("--stage", "all")
    & $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
    Note "STEP $($s.tag) exit $LASTEXITCODE"
    # push each step as it lands rather than holding everything to the end
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") `
        -OnBusy Skip -Tag "crossgen/$($s.tag)"
}
Note "DONE"
