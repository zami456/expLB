$ErrorActionPreference = "Continue"
$repo = "C:\Users\user4\expLB"
Set-Location $repo
$env:PYTHONIOENCODING = "utf8"
$pyAN = Join-Path $repo ".venv-analysis\Scripts\python.exe"
$SRC  = Join-Path $repo "src\assistant_axis_ndif"
$status = Join-Path $repo "logs\cap_ctrl_layers.status"
function Note($m) { "$((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))  $m" | Out-File -Append -Encoding utf8 $status }
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $status
# F-89's stated caveat: the format-matched control had only 3 layers while every other kd11
# bank has 21. L32 is the measured peak for cap_s23, so the control MUST cover it or the
# comparison is made at a layer neither side peaks at.
$DS = @("instructed-deception","harm-pressure-choice","harm-pressure-knowledge-report",
        "harm-pressure-kr-matched","convincing-game","insider-trading","ai_liar",
        "sandbagging_v2","roleplaying","mask","sycophancy_mt")
$F = @("raw","random","length","datapca","random@2","datapca@2",
       "pc:cap_vs_neutral","pc:cap_vs_para","pc:cap_only_ctrl",
       "pc:cap_vs_neutral@2","pc:cap_vs_para@2","pc:cap_only_ctrl@2")
$L = @(0,4,8,12,16,22,24,28,32,36,48,52,56,60,64,68,72,76)
Note "crossgen cap_ctrl on 18 missing layers (20/40/44 already done)"
& $pyAN "$SRC\27_crossgen.py" --run_dir "run-coords\passB" `
    --out_root "run-crossgen\kd11\banks\cap_ctrl" --bank_name "cap_ctrl" `
    --bank "run-cap-ctrl\axes_s23.pt" --baseline_root "run-crossgen\kd11" `
    --datasets @DS --features @F --extra_features @F --extra_probes paper `
    --layers @L --stage all *>> (Join-Path $repo "logs\cap_ctrl_layers.log")
# COUNT THE ARM THE STEP ACTUALLY PRODUCES. The first version counted every .json in
# matrices/, which on 2026-09-21 measured 42 against a prediction of 21 and logged PASS --
# correct by luck, because the crossgen writes BOTH mean_* and last_* and the two happened
# to be complete. A short mean arm padded by a full last arm would also have passed. Gate on
# the mean files by name. Fifth loose gate of the day.
$mm = @(Get-ChildItem "run-crossgen\kd11anks\cap_ctrl\matrices\mean_L*_s42.json" -EA SilentlyContinue).Count
$ml = @(Get-ChildItem "run-crossgen\kd11anks\cap_ctrl\matrices\last_L*_s42.json" -EA SilentlyContinue).Count
Note ("measured={0} mean matrices ({1} last)  predicted=21 mean  {2}" -f $mm, $ml, $(if ($mm -eq 21) { "PASS" } else { "GATE FAILED" }))
Note "DONE"
