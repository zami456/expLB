# READ-ONLY: classify the loose root-level .py scripts on the lab PC (verify batch B, item AN4).
# For each .py directly under C:\Users\user4\expLB with mtime 2026-09-19..2026-09-21, report
# whether it writes anything (file writes / plots / torch.save) or is print-only.
$ErrorActionPreference = 'Stop'
$d = 'C:\Users\user4\expLB'
$wpat = "open\([^)]*['""][wax]|to_csv|to_json|json\.dump|savefig|torch\.save|np\.save|write_text|\.write\(|makedirs|mkdir\("
$n = 0; $pw = 0
Get-ChildItem -LiteralPath $d -Filter *.py -File |
  Where-Object { $_.LastWriteTime -ge (Get-Date '2026-09-19') -and $_.LastWriteTime -lt (Get-Date '2026-09-22') } |
  Sort-Object LastWriteTime |
  ForEach-Object {
    $c = Get-Content -LiteralPath $_.FullName -Raw
    $w = [regex]::IsMatch($c, $wpat)
    $n++
    if (-not $w) { $pw++ }
    '{0,-24} {1:yyyy-MM-dd HH:mm} {2,7}B writes={3}' -f $_.Name, $_.LastWriteTime, $_.Length, $w
  }
"TOTAL in window = $n ; print-only (no write pattern) = $pw"
