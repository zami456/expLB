<#
  The paper's own tests, at the paper's own two model sizes.

  Mahadik & Skapars evaluate persona coordinates three ways (section 5, figures 3-7 and 9):
    1. ZERO-SHOT  -- contrast direction and PC1 projected straight onto dataset activations;
    2. TRAINED    -- logistic probes on the top-k persona-PC projections vs on raw activations,
                     trained on one dataset and tested on every other (the off-diagonal);
    3. CONTROLS   -- the same probes on a random k-dim subspace and on dataset-specific PCA at
                     matched k, which is what separates "the axis helps" from "any low-dim
                     projection helps".
  `27_crossgen.py` produces all three in one pass, so this script is a thin wrapper.

  k=3 MATCHES THE PAPER. Its learned-probe experiments "train on the full training split of each
  source dataset projected onto the first 3 persona PCs" (section 4). Our own runs default to
  k=5, and F-39 found the matched-control result is k-DEPENDENT -- the axis beats dataset PCA by
  +0.138 at k=2 and by +0.004 at k=5 -- so using the paper's k is not a detail.

  DATASETS ARE THE PAPER'S TABLE 1 DECEPTION SET, under our names: AILiar, Roleplaying,
  ConvGame, InstrDec, HarmPressureChoice. The other six datasets in our grid have no counterpart
  in the paper and would not make the comparison sharper.

  CPU ONLY. Reads stored activations; never imports nnsight.

  Written as a FILE rather than an inline ssh command because interpolating a loop variable into
  a path through ssh -> powershell -> argument array silently produced no output twice today:
  the redirect target became a literal and the run vanished with exit 0.
#>
param([string[]]$Rungs = @("8b", "3b"), [int]$K = 3)

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC = Join-Path $repo "src\assistant_axis_ndif"
$logDir = Join-Path $repo "logs"
$status = Join-Path $logDir "scale_crossgen.status"

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Append -Encoding utf8 $status

# CORRECTED 2026-09-20 23:35 (F-60). The paper's TABLE 1 lists HarmPressureChoice, but every
# FIGURE evaluates "ConvGame InstrDec Mask AILiar Roleplay" -- MASK appears 33 times in the
# paper and has no row in Table 1, while HarmPressureChoice appears twice, both inside Table 1
# and in no figure. The figures are what the claims rest on, so MASK is the correct fifth
# dataset. Using HPC also cost us the one column that came through 5:1 class-imbalanced (F-56).
$DS = @("instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying")

foreach ($rung in $Rungs) {
    $coords = Join-Path $repo "run-scale\$rung\coords"
    $bank = Join-Path $repo "run-scale\$rung\axes_$rung.pt"
    $out = Join-Path $repo "run-crossgen\scale_$rung"
    foreach ($p in @($coords, $bank, $py)) {
        if (-not (Test-Path $p)) { Note "$rung MISSING $p -- skipping"; continue }
    }
    Note "$rung crossgen start  k=$K  $($DS.Count) datasets"

    $a = @("$SRC\27_crossgen.py", "--run_dir", $coords, "--out_root", $out,
           "--bank_name", "paper16_$rung", "--bank", $bank, "--datasets") + $DS + @(
           "--k", "$K", "--stage", "all")
    # NO --features override. The script's default list is exactly the paper's comparison set:
    #   raw, random, datapca (the two controls Figure 9A reports), length, plus the persona
    #   features and their @2 variants. Passing an explicit list on the first run silently
    #   dropped raw and both controls, which are the whole point -- a persona feature's number
    #   means nothing without the baseline it is supposed to beat.
    & $py @a 2>&1 | ForEach-Object { "$_" } |
        Out-File -Append -Encoding utf8 (Join-Path $logDir "crossgen_scale_$rung.log")
    $n = @(Get-ChildItem (Join-Path $out "matrices") -Filter "*.json" -EA SilentlyContinue).Count
    Note "$rung crossgen exit ${LASTEXITCODE}: $n matrices"
}
Note "DONE"
