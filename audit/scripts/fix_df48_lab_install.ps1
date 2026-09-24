# DF-48: install the patched run_scale_chain.ps1 on the lab PC.
# Owner-authorised 2026-09-23 ("solve this - the lab copy still needs the DF-48 patch").
# Same procedure as the 2026-09-22 guard-bug fix: verify the target is the expected original,
# back it up inside audit_scratch, install, verify the new hash, print everything.
$ErrorActionPreference = 'Stop'
$target = 'C:\Users\user4\expLB\scale\scripts\run_scale_chain.ps1'
$staged = 'C:\Users\user4\audit_scratch\scripts\run_scale_chain.ps1'
$backup = 'C:\Users\user4\audit_scratch\backup_DF48_run_scale_chain.ps1'
$expectBefore = '__EXPECT_BEFORE__'
$expectAfter  = '__EXPECT_AFTER__'

function H($p) { if (Test-Path -LiteralPath $p) { (Get-FileHash -LiteralPath $p -Algorithm MD5).Hash.ToLower() } else { 'MISSING' } }

$before = H $target
"target  before : $before"
"staged         : $(H $staged)"

if ($before -ne $expectBefore) {
    "ABORT: target is not the expected pre-fix original ($expectBefore). Nothing written."
    exit 2
}
if ((H $staged) -ne $expectAfter) {
    "ABORT: staged file is not the expected patched version ($expectAfter). Nothing written."
    exit 3
}

Copy-Item -LiteralPath $target -Destination $backup -Force
"backup written : $backup  ($(H $backup))"

Copy-Item -LiteralPath $staged -Destination $target -Force
$after = H $target
"target  after  : $after"

if ($after -eq $expectAfter) { "RESULT: OK - lab copy now matches the patched repo copy" }
else { "RESULT: MISMATCH - expected $expectAfter" ; exit 4 }

# show the patched block so the change is visible in the log
"---- lines 195-215 of the installed file ----"
Get-Content -LiteralPath $target | Select-Object -Skip 194 -First 21
