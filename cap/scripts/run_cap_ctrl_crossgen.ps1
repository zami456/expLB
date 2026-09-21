$ErrorActionPreference="Continue"
$repo="C:\Users\user4\expLB"; Set-Location $repo
$env:PYTHONIOENCODING="utf8"
$pyAN=Join-Path $repo ".venv-analysis\Scripts\python.exe"
$status=Join-Path $repo "logs\cap_controls.status"
function Note($m){ "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
Note "4 axes  measured=3 bases, min cos(PC1,contrast)=0.936  predicted=3 bases, >=0.90  PASS (re-run after the `$src/`$SRC collision)"
$DS=@("instructed-deception","harm-pressure-choice","harm-pressure-knowledge-report","harm-pressure-kr-matched","convincing-game","insider-trading","ai_liar","sandbagging_v2","roleplaying","mask","sycophancy_mt")
$F=@("raw","random","length","datapca","random@2","datapca@2","pc:cap_vs_neutral","pc:cap_vs_para","pc:cap_only_ctrl","pc:cap_vs_neutral@2","pc:cap_vs_para@2","pc:cap_only_ctrl@2")
Note "5 crossgen on the 11-dataset grid, layers 20/40/44"
$t0=Get-Date
& $pyAN (Join-Path $repo "src\assistant_axis_ndif\27_crossgen.py") --run_dir "run-coords\passB" `
  --out_root "run-crossgen\kd11\banks\cap_ctrl" --bank_name "cap_ctrl" `
  --bank "run-cap-ctrl\axes_s23.pt" --baseline_root "run-crossgen\kd11" `
  --datasets @DS --features @F --extra_features @F --extra_probes paper `
  --layers 20 40 44 --stage all *>> (Join-Path $repo "logs\cap_ctrl_crossgen.log")
$m=@(Get-ChildItem "run-crossgen\kd11\banks\cap_ctrl\matrices\*.json" -EA SilentlyContinue).Count
Note ("5 crossgen  measured={0} matrices  predicted=>=6  {1}  {2} min" -f $m,$(if($m -ge 6){"PASS"}else{"GATE FAILED"}),[math]::Round(((Get-Date)-$t0).TotalMinutes,1))
Note "DONE"
