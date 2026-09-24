# READ-ONLY: progress of the RF-16 like-for-like sweep.
$log = 'C:\Users\user4\audit_scratch\out\rf16_crossgen_scale_70b_paper16fp32.log'
$out = 'C:\Users\user4\expLB\run-crossgen\scale_70b_paper16fp32'
if (Test-Path $log) { "log " + (Get-Item $log).Length + " B, last write " + (Get-Item $log).LastWriteTime.ToString('HH:mm:ss'); Get-Content $log -Tail 6 } else { "no log yet" }
if (Test-Path "$out\matrices") { "matrices: " + @(Get-ChildItem "$out\matrices" -Filter *.json).Count } else { "no matrices dir yet" }
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object { "python PID " + $_.ProcessId + " started " + $_.CreationDate.ToString('HH:mm:ss') }
