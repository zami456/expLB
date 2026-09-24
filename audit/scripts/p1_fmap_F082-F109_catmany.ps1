# READ-ONLY. Print several small lab files (names relative to C:\Users\user4\expLB, comma-separated), each with a header
# (bytes, mtime, md5). Refuses files > 64 KB. Sent via -EncodedCommand; writes nothing on the lab PC.
# Usage: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F082-F109_catmany.ps1 "FILES=a.py,b.py"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
foreach ($n in '__FILES__'.Split(',')) {
  $p = Join-Path 'C:\Users\user4\expLB' $n
  if (-not (Test-Path -LiteralPath $p)) { "##### $n  NOT FOUND"; continue }
  $i = Get-Item -LiteralPath $p
  if ($i.Length -gt 65536) { "##### $n  too big ($($i.Length))"; continue }
  $h = (Get-FileHash -LiteralPath $p -Algorithm MD5).Hash.ToLower()
  "##### $n  bytes=$($i.Length) mtime=$($i.LastWriteTime.ToString('yyyy-MM-dd HH:mm:ss')) md5=$h"
  Get-Content -LiteralPath $p -Raw
}
