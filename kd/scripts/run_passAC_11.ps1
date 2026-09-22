<#
  The passA and passC grids re-run with the ELEVENTH dataset, now that it exists there.

  WHY NOW. run_kd_passAC.ps1 runs TEN datasets and says so at length: harm-pressure-kr-matched
  was excluded from passA and passC because those stores held a 200-row stratified subsample of
  HP-KR whose matched view was 98 rows at 89 deceptive / 9 honest, against passB's 124 / 134. An
  AUROC against nine negatives is noise with a number attached. That reasoning was correct and is
  now obsolete: HPKR-Topup extracted the full 2,139 rows into both stores on 2026-09-21, and all
  three now carry the same 258-row slice at 124 / 134 (F-69, F-71).

  WHAT IT BUYS. passA stores all 80 layers, so this produces the first LAYER PROFILE for
  harm-pressure-kr-matched -- the column carrying F-03, the strongest positive result in the
  project, which has only ever been measured at three layers on passB. F-34 showed layer choice
  is not cosmetic (a factor probe moved 0.784 -> 0.509 between L20 and L40), so a headline quoted
  at one layer without a profile carries an unquantified degree of freedom. passC is the only
  per-token store, so it gives the pooling check on the same column.

  A NEW OUTPUT ROOT, NOT --recompute. run-crossgen\kd11\passA already holds 10x10 matrices that
  findings cite. Re-running there would either skip them (27_crossgen.py resumes by matrix file)
  or, with --recompute, silently replace a published 10-dataset result with an 11-dataset one of
  a different shape. passA11 / passC11 keeps both and makes the 10-vs-11 comparison possible.

  ASYMMETRY TO REMEMBER WHEN READING THE OUTPUT. passA and passC still hold 200-row subsamples of
  the other nine datasets. HP-KR and its matched slice are full; the rest are not. That is fine
  for reading the matched column down the layer axis, which is the point, and it is NOT a
  like-for-like grid against passB.

  Scheduled Task only. CPU, no NDIF -- the activations are already stored.
#>
param([switch]$Force)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$pyAA = Join-Path $repo ".venv-aa\Scripts\python.exe"
$SRCDIR = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "passAC11.status"
$log = Join-Path $logDir "passAC11.log"
$bank = Join-Path $repo "run-kd\axes_vectors_s23.pt"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

if ((Test-Path $status) -and -not $Force -and (Select-String -Path $status -Pattern '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+DONE\s*$' -Quiet)) {
    Note "SKIP already DONE; -Force to re-run"; exit 0
}
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status
if (-not (Test-Path $bank)) { Note "MISSING $bank -- stopping"; exit 1 }

# ---- STEP 0: BUILD THE VIEW ----------------------------------------------------------
# harm-pressure-kr-matched is NOT a registered dataset -- eval_datasets.load_dataset raises
# KeyError on it. It is a VIEW: a directory holding a filtered rows.jsonl plus hardlinks to
# HP-KR's own chunks, which load_dataset then reads by row_id, skipping any chunk row absent
# from rows.jsonl. passB has had one since the 11-dataset grid; passA and passC did not.
# Running the grid without this step fails every matched cell.
Note "STEP 0 build the kr-matched view in passA and passC"
& $pyAA (Join-Path $repo "make_view_ac.py") *>> (Join-Path $logDir "passAC11_view.log")

# Build every path segment with Join-Path, never by interpolating separators into a string.
# The first version of this gate read 0 rows from a view that had been written CORRECTLY (258
# rows at 124/134, confirmed in passAC11_view.log). The literal ended "kr-matched" + backslash +
# "rows.jsonl"; the backslash-r was consumed as a carriage return, so what Test-Path received was
# "kr-matchedows.jsonl", which quite reasonably did not exist. It reported GATE FAILED on sound
# work -- the mirror image of the two gates that reported PASS on nothing. Segment-wise
# Join-Path cannot express that bug.
foreach ($p in @("passA", "passC")) {
    $vdir = Join-Path (Join-Path (Join-Path $repo "run-coords") $p) "harm-pressure-kr-matched"
    $rj = Join-Path $vdir "rows.jsonl"
    $n = if (Test-Path $rj) { @(Get-Content $rj).Count } else { 0 }
    $nc = @(Get-ChildItem $vdir -Filter "chunk_*.pt" -EA SilentlyContinue).Count
    Note ("STEP 0/{0}  measured={1} matched rows, {2} chunks  predicted=258  {3}" -f
          $p, $n, $nc, $(if ($n -eq 258 -and $nc -gt 0) { "PASS" } else { "GATE FAILED" }))
}

# ELEVEN. The eleventh is harm-pressure-kr-matched, which F-20 excluded and F-71 restored.
$DS = @("instructed-deception", "harm-pressure-choice", "harm-pressure-knowledge-report",
        "harm-pressure-kr-matched", "convincing-game", "insider-trading", "ai_liar",
        "sandbagging_v2", "roleplaying", "mask", "sycophancy_mt")

# Peak layers first so an interruption still leaves the numbers every conclusion is read at;
# then the full sweep, which is the layer profile and the whole reason passA exists.
# An empty layers list means "every stored layer" (27_crossgen.py --layers default=[]).
$steps = @(
    @{ tag = "A-root";     run = "run-coords\passA"; out = "run-crossgen\kd11\passA11"; layers = @("40", "44", "20"); bank = $false },
    @{ tag = "A-kd";       run = "run-coords\passA"; out = "run-crossgen\kd11\passA11\banks\kd_s23"; layers = @("40", "44", "20"); bank = $true },
    @{ tag = "C-root";     run = "run-coords\passC"; out = "run-crossgen\kd11\passC11"; layers = @(); bank = $false },
    @{ tag = "C-kd";       run = "run-coords\passC"; out = "run-crossgen\kd11\passC11\banks\kd_s23"; layers = @(); bank = $true },
    @{ tag = "A-root-all"; run = "run-coords\passA"; out = "run-crossgen\kd11\passA11"; layers = @(); bank = $false },
    @{ tag = "A-kd-all";   run = "run-coords\passA"; out = "run-crossgen\kd11\passA11\banks\kd_s23"; layers = @(); bank = $true }
)

foreach ($step in $steps) {
    $t0 = Get-Date
    Note "STEP $($step.tag) layers=$($step.layers -join ',') start"
    $a = @("$SRCDIR\27_crossgen.py", "--run_dir", $step.run, "--out_root", $step.out,
           "--datasets") + $DS
    # Omit --layers entirely when the list is empty. A bare switch with no value swallows the
    # NEXT argument -- the hazard that silently killed two crossgen steps on 2026-09-19 and made
    # the chain report DONE having run half its work.
    if ($step.layers.Count) { $a += @("--layers") + $step.layers }
    if ($step.bank) {
        $a += @("--bank_name", "kd_s23", "--bank", $bank,
                "--baseline_root", (Split-Path (Split-Path $step.out -Parent) -Parent),
                "--features", "pc:deception", "axes", "axes_resid", "pc:unified",
                              "pc:knowledge_denial", "pc:kd_assertion",
                "--extra_features", "axes", "pc:knowledge_denial", "pc:kd_assertion")
    }
    $a += @("--stage", "all")
    & $pyAN @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log

    # THE GATE ASSERTS THE ELEVENTH DATASET IS ACTUALLY IN THE MATRIX, not merely that files
    # exist. Two gates passed today while measuring nothing -- one compared against the wrong
    # number, one only checked that a file existed. The whole point of this run is the eleventh
    # column, so a matrix that silently came out 10-wide is the failure worth catching.
    $mdir = Join-Path $repo (Join-Path $step.out "matrices")
    $files = @(Get-ChildItem $mdir -Filter "*.json" -EA SilentlyContinue)
    $nds = -1
    if ($files.Count) {
        try { $nds = (Get-Content $files[0].FullName -Raw | ConvertFrom-Json).datasets.Count }
        catch { $nds = -1 }
    }
    $mins = [math]::Round(((Get-Date) - $t0).TotalMinutes, 1)
    $ok = ($LASTEXITCODE -eq 0) -and ($files.Count -gt 0) -and ($nds -eq 11)
    Note ("STEP {0} exit {1}  matrices={2}  datasets_in_matrix={3} (predicted 11)  {4} min  {5}" -f
          $step.tag, $LASTEXITCODE, $files.Count, $nds, $mins, $(if ($ok) { "PASS" } else { "GATE FAILED" }))
}

& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $D "sync_kd.ps1") `
    -OnBusy Wait -Tag "passAC11/final" -EA SilentlyContinue
Note "DONE"
