# One line for rpb_monitor.sh: what of the random-bank run is alive right now.
#   queue=<n> chain=<n> python=<n> task=<state> freeMB=<n>
# queue  = the RPB-queue powershell; chain = run_chain / k_chain / alpaca_chain powershells of this run;
# python = python processes of this run (each shard is a venv launcher + child, so 2 per shard).
$p = @(Get-CimInstance Win32_Process -Filter "Name='powershell.exe' or Name='python.exe'" -EA SilentlyContinue)
$q = @($p | Where-Object { $_.CommandLine -match '_queue\\queue\.ps1' }).Count
$c = @($p | Where-Object { $_.Name -eq 'powershell.exe' -and $_.CommandLine -match 'random_persona_bank\\results\\[^\\]+\\code\\run_chain\.ps1|k_chain\.ps1.*random_persona_bank|alpaca_chain\.ps1.*random_persona_bank' }).Count
$y = @($p | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'random_persona_bank' }).Count
$t = try { (Get-ScheduledTask -TaskName RPB-queue -EA Stop).State } catch { "MISSING" }
$m = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1024)
"queue=$q chain=$c python=$y task=$t freeMB=$m"
