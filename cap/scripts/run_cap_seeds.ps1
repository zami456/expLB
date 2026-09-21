$ErrorActionPreference="Continue"
$repo="C:\Users\user4\expLB"; Set-Location $repo
$env:PYTHONIOENCODING="utf8"
$py=Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC=Join-Path $repo "src\assistant_axis_ndif"
$log=Join-Path $repo "logs\cap_seeds.log"
$status=Join-Path $repo "logs\cap_seeds.status"
function Note($m){ "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status
$DS=@("instructed-deception","harm-pressure-choice","harm-pressure-knowledge-report","harm-pressure-kr-matched","convincing-game","insider-trading","ai_liar","sandbagging_v2","roleplaying","mask","sycophancy_mt")
$F=@("raw","random","length","datapca","random@2","datapca@2","pc:capitulation","pc:cap_vs_default","pc:cap_only","pc:capitulation@2","pc:cap_vs_default@2","pc:cap_only@2")
# Seed robustness for the cap result. Our strongest NEGATIVE (pooled training) is checked on
# five seeds; our strongest POSITIVE (the cap bank) is on one. Same layer L40, same 11 datasets.
$t0=Get-Date
& $py "$SRC\27_crossgen.py" --run_dir "run-coords\passB" --out_root "run-crossgen\kd11\banks\cap_s23" `
  --bank_name "cap_s23" --bank "run-cap\axes_s23.pt" --baseline_root "run-crossgen\kd11" `
  --datasets @DS --features @F --extra_features @F --extra_probes paper --layers 40 --main_seed 42 --extra_seeds 1 7 13 --stage all `
  *>> $log
$n=@(Get-ChildItem "run-crossgen\kd11\banks\cap_s23\matrices\mean_L40_s*.json" -EA SilentlyContinue).Count
Note ("cap seeds exit {0}  L40 seed-matrices={1} (predicted 4)  {2} min  {3}" -f $LASTEXITCODE,$n,
      [math]::Round(((Get-Date)-$t0).TotalMinutes,1), $(if($n -ge 4){"PASS"}else{"GATE FAILED"}))
Note "DONE"
