# W6G: read-only excerpts from lab-PC logs for claims S-05/S-06, S-26, S-31, S-24/S-25, S-19, S-21.
# Read-only: Get-Content / Select-String only. No writes.
$ErrorActionPreference = 'Continue'
$L = 'C:\Users\user4\expLB\logs'

function Show($name, $pattern, $ctxBefore, $ctxAfter, $max) {
  $p = Join-Path $L $name
  Write-Output ("#### " + $name)
  if (-not (Test-Path $p)) { Write-Output "  MISSING"; return }
  $fi = Get-Item $p
  Write-Output ("  bytes=" + $fi.Length + " mtime=" + $fi.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss'))
  try {
    $hits = Select-String -Path $p -Pattern $pattern -Context $ctxBefore,$ctxAfter -ErrorAction Stop | Select-Object -First $max
    foreach ($h in $hits) {
      foreach ($l in $h.Context.PreContext) { Write-Output ("    | " + $l) }
      Write-Output ("  >>| " + $h.Line)
      foreach ($l in $h.Context.PostContext) { Write-Output ("    | " + $l) }
      Write-Output "  ---"
    }
    if (-not $hits) { Write-Output "  NO MATCH" }
  } catch { Write-Output ("  ERR " + $_.Exception.Message) }
}

Show 'q8_smoke.log'      'self-check|layer 44 --|rows \d+, sampled tokens' 0 24 4
Show 'q8_full.log'       'self-check' 0 0 8
Show 'kd_axes.log'       'PC1 explains|cos\(PC1, assistant axis\)' 2 2 6
Show 'scale_8b_axes.log' 'PC1 explains|report at layer' 0 22 6
Show 'cap_axes_all.log'  'cos\(capitulation|max\|A-B\|' 2 6 6
Show 'cap_axes_s23.log'  'cos\(capitulation|max\|A-B\|' 2 6 6
