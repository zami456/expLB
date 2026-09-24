# Owner-authorised fix for RF-16 (2026-09-22), run ON the lab PC.
# Re-runs the owner's own 27_crossgen.py exactly as the 09-22 70B k=3 sweep (passB activations, the 5 datasets of the
# paper, k=3, all stored layers, mean+last pooling) but with the paper16 fp32 bank instead of the 36-persona kd bank,
# so the 70B rung is like-for-like with the 3B/8B paper16 rungs. CPU only (no NDIF). Writes ONLY to a NEW folder
# run-crossgen\scale_70b_paper16fp32 (does not exist before this run). Log in audit_scratch\out\.
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:PYTHONIOENCODING = 'utf-8'
Set-Location 'C:\Users\user4\audit_scratch'
$repo = 'C:\Users\user4\expLB'
$py = "$repo\.venv-analysis\Scripts\python.exe"
$out = "$repo\run-crossgen\scale_70b_paper16fp32"
$log = 'C:\Users\user4\audit_scratch\out\rf16_crossgen_scale_70b_paper16fp32.log'
if (Test-Path -LiteralPath $out) { "REFUSING: $out already exists"; exit 1 }
$a = @("$repo\src\assistant_axis_ndif\27_crossgen.py",
       "--run_dir", "$repo\run-coords\passB",
       "--out_root", $out,
       "--bank_name", "paper16_70b_fp32",
       "--bank", "$repo\run-paper16\axes_vectors_s3_fp32.pt",
       "--datasets", "instructed-deception", "mask", "convincing-game", "ai_liar", "roleplaying",
       "--k", "3", "--stage", "all")
"START $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Encoding utf8 $log
& $py @a 2>&1 | ForEach-Object { "$_" } | Out-File -Append -Encoding utf8 $log
"EXIT $LASTEXITCODE $((Get-Date).ToString('yyyy-MM-dd HH:mm:ss'))" | Out-File -Append -Encoding utf8 $log
"exit $LASTEXITCODE; matrices: " + @(Get-ChildItem "$out\matrices" -Filter *.json -EA SilentlyContinue).Count
