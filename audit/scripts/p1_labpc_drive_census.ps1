# Phase 1 census of Google Drive tamara:expLB, run ON the lab PC with its own rclone (read-only: lsjson).
# Writes ONLY to C:\Users\user4\audit_scratch\census\drive_lsjson.json
$rc = 'C:\Users\user4\expLB\tools\rclone.exe'
if (-not (Test-Path -LiteralPath $rc)) { $rc = 'C:\Users\user4\tools\rclone.exe' }
$out = 'C:\Users\user4\audit_scratch\census\drive_lsjson.json'
$err = 'C:\Users\user4\audit_scratch\census\drive_lsjson.err'
& $rc lsjson -R --hash --files-only 'tamara:expLB' 1> $out 2> $err
"rclone exit $LASTEXITCODE; bytes " + (Get-Item -LiteralPath $out).Length
