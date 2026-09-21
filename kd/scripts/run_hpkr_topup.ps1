<#
  Top up harm-pressure-knowledge-report to its FULL 2,139 rows in passA and passC.

  WHY. Both stores kept only a 200-row label-stratified subsample of HP-KR. HP-KR's honest class
  is overwhelmingly "I know the answer" -- assertions, which by construction cannot appear in a
  DENIAL-matched slice -- so only 9 honest rows survived into the matched slice, against 89
  deceptive (F-20). An AUROC against nine negatives is noise with a number attached, which is why
  the passA/passC grids run with ten datasets instead of eleven.

  This creates NO new rows. It stops discarding ones that already exist. What it buys is the
  matched slice on the two stores that have something passB does not: passA is all 80 layers (a
  LAYER PROFILE for F-03, the project's strongest positive result, which currently has none), and
  passC is the only per-token store (a POOLING check on the same).

  --overwrite IS REQUIRED AND IS NOT DESTRUCTIVE HERE. Read 8_coords_ndif.py:552 before changing
  it: --overwrite has exactly one effect, rebuilding items.pt, and nothing in that script ever
  deletes a chunk. Without it the cached 200-row items.pt is reloaded and the run reports
  "already complete" having done nothing -- the same stale-cache shape as F-44 and F-47. With it,
  items.pt is rebuilt to all 2,139 spans, resume scans the existing chunks for row_ids already
  done, and only the ~1,939 missing rows are extracted. Re-running after an interruption is free
  and idempotent.

  SETTINGS ARE READ OFF THE EXISTING CHUNKS, not assumed: passA is --layers all (80) and passC is
  --layers 20 22 28 40 44 48 --per_token, both with the default pooling (mean, last) and the
  default response span. New chunks must match or the store cannot be stacked.

  Scheduled Task, never Start-Process or a long SSH command -- OpenSSH takes the remote process
  tree down with the session, which killed a 15-minute job on 2026-09-21 that happened to have
  finished 12 seconds earlier.
#>
param([string]$Jobs = "A C")

$ErrorActionPreference = "Continue"
$D = $PSScriptRoot
$repo = Split-Path (Split-Path $D -Parent) -Parent
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"

$py = Join-Path $repo ".venv-aa\Scripts\python.exe"
$script = Join-Path $repo "src\assistant_axis_ndif\8_coords_ndif.py"
$logDir = Join-Path $repo "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$status = Join-Path $logDir "hpkr_topup.status"
$DS = "harm-pressure-knowledge-report"
$EXPECT = 2139

function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }

function Count-Rows($store) {
    # The number that matters is DISTINCT row_ids actually on disk in chunks, not the line count
    # of rows.jsonl -- rows.jsonl is rewritten from items.pt at the END of a successful pass and
    # would read 2,139 even if extraction had covered a fraction of them.
    $d = Join-Path $repo "run-coords\$store\$DS"
    $expr = "import torch,glob,sys;" +
            "ids=set();" +
            "[ids.update(torch.load(c,weights_only=False)['row_ids']) for c in glob.glob(r'$d\chunk_*.pt')];" +
            "print(len(ids))"
    $n = & $py -c $expr 2>$null
    if ($LASTEXITCODE -ne 0 -or -not $n) { return -1 }
    return [int]($n.Trim())
}

"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  jobs=$Jobs" | Out-File -Append -Encoding utf8 $status

$before = @{}
foreach ($s in @("passA", "passC")) { $before[$s] = Count-Rows $s }
Note "before: passA=$($before['passA'])  passC=$($before['passC'])  target=$EXPECT"

if ($Jobs -match "A") {
    Note "passA: $DS at all 80 layers (adds ~$($EXPECT - $before['passA']) rows, ~2.6 MB/row => ~5 GB)"
    & $py $script --datasets $DS --layers all --overwrite `
        --output_dir "run-coords\passA" *>> (Join-Path $logDir "hpkr_topup_passA.log")
    $n = Count-Rows "passA"
    Note ("passA  measured={0} rows  predicted={1}  {2}" -f $n, $EXPECT,
          $(if ($n -eq $EXPECT) { "PASS" } else { "GATE FAILED" }))
}

if ($Jobs -match "C") {
    Note "passC: $DS per-token at 6 layers (adds ~$($EXPECT - $before['passC']) rows, ~0.8 MB/row => ~1.6 GB)"
    & $py $script --datasets $DS --layers 20 22 28 40 44 48 --per_token --overwrite `
        --output_dir "run-coords\passC" *>> (Join-Path $logDir "hpkr_topup_passC.log")
    $n = Count-Rows "passC"
    Note ("passC  measured={0} rows  predicted={1}  {2}" -f $n, $EXPECT,
          $(if ($n -eq $EXPECT) { "PASS" } else { "GATE FAILED" }))
}

# Free space is worth recording next to the row counts: this run is the largest single write the
# project has made to C:, on a machine shared with another user's training jobs.
Note ("C: {0:N1} GB free" -f ((Get-PSDrive C).Free / 1GB))

$rc = Join-Path $repo "tools\rclone.exe"
if (-not (Test-Path $rc)) { $rc = "C:\Users\user4\tools\rclone.exe" }
if (Test-Path $rc) {
    foreach ($p in @("passA", "passC")) {
        & $rc copy (Join-Path $repo "run-coords\$p\$DS") "tamara:expLB/coords/$p/$DS" `
            --include "*.pt" --include "*.jsonl" --checksum --transfers 4 `
            --log-file (Join-Path $logDir "hpkr_topup_rclone.log") --log-level INFO
        Note "rclone $p exit $LASTEXITCODE"
    }
}

Note "DONE"
