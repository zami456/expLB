# READ-ONLY test of the proposed RF-14 guard pattern against real status files on the lab PC.
# Expect: OLD pattern '^DONE' = False everywhere; NEW pattern = True for finished chains.
$old = '^DONE'
$new = '^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\s+DONE\s*$'
foreach ($f in 'falsify.status', 'premise_layers.status', 'premise_extras.status', 'cap_layers.status', 'cap_controls.status', 'sarc_pilot.status', 'passAC11.status', 'kd_sync.status') {
  $p = "C:\Users\user4\expLB\logs\$f"
  if (-not (Test-Path -LiteralPath $p)) { "$f : missing"; continue }
  $o = [bool](Select-String -LiteralPath $p -Pattern $old -Quiet)
  $n = [bool](Select-String -LiteralPath $p -Pattern $new -Quiet)
  $last = (Get-Content -LiteralPath $p -Tail 1)
  "{0,-24} old={1,-5} new={2,-5} last line: {3}" -f $f, $o, $n, $last
}
