# Phase 0, READ-ONLY: print the exact rclone invocations in the periodic task scripts.
$files = @{ 'night_chain.ps1'=@(44,55); 'run_passC.ps1'=@(33,39) ; 'sync_crossgen.ps1'=@(94,100); 'sync_kd.ps1'=@(52,82); 'coords_supervisor.ps1'=@(124,140) }
$paths = @{ 'night_chain.ps1'='C:\Users\user4\night_chain.ps1'; 'run_passC.ps1'='C:\Users\user4\run_passC.ps1'; 'sync_crossgen.ps1'='C:\Users\user4\expLB\src\assistant_axis_ndif\sync_crossgen.ps1'; 'sync_kd.ps1'='C:\Users\user4\expLB\kd\scripts\sync_kd.ps1'; 'coords_supervisor.ps1'='C:\Users\user4\expLB\src\assistant_axis_ndif\coords_supervisor.ps1' }
foreach ($k in $files.Keys) { "=== $k"; $c = Get-Content -LiteralPath $paths[$k]; $r = $files[$k]; for ($i=$r[0]; $i -le [Math]::Min($r[1],$c.Count); $i++) { "{0,4}: {1}" -f $i, $c[$i-1] } }
