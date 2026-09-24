# Owner-authorised backup (RF-10), run ON the lab PC. COPY ONLY into a NEW Drive folder; never sync/move/delete.
# --immutable: rclone refuses to modify any file that already exists at the destination (none should — new folder).
# Then `rclone check --one-way` verifies every listed file by hash. Logs go to audit_scratch\out\.
$rc = 'C:\Users\user4\expLB\tools\rclone.exe'
if (-not (Test-Path -LiteralPath $rc)) { $rc = 'C:\Users\user4\tools\rclone.exe' }
$dest = 'tamara:expLB/_labpc_only_backup_2026-09-22'
$L = 'C:\Users\user4\audit_scratch\out'
$jobs = @(
  @('C:\Users\user4\expLB', 'expLB'),
  @('D:\expLB-acts', 'expLB-acts'),
  @('C:\Users\user4\expLBactivationLAYER22', 'expLBactivationLAYER22'),
  @('C:\Users\user4', 'user4-home-loose-files')
)
foreach ($j in $jobs) {
  $src = $j[0]; $lab = $j[1]; $list = "$L\backup_files_$lab.txt"
  & $rc copy $src "$dest/$lab" --files-from-raw $list --immutable --transfers 4 --checkers 8 --log-file "$L\backup_copy_$lab.log" --log-level INFO
  "copy $lab exit $LASTEXITCODE"
  & $rc check $src "$dest/$lab" --files-from-raw $list --one-way --log-file "$L\backup_check_$lab.log" --log-level INFO
  "check $lab exit $LASTEXITCODE (0 = every listed file present on Drive with identical hash)"
}
& $rc size $dest --json
"END"
