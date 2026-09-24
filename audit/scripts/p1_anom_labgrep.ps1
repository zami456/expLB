# READ-ONLY lab search: Select-String over the loose lab scripts in C:/Users/user4/expLB and C:/Users/user4 for orphan output names (Task B).
# Run: python audit/scripts/labpc_run.py audit/scripts/p1_anom_labgrep.ps1
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$files = @(Get-ChildItem -LiteralPath 'C:\Users\user4\expLB' -File -Include *.py,*.ps1 -Name) 
$files = @(Get-ChildItem -Path 'C:\Users\user4\expLB\*' -File -Include *.py,*.ps1) + @(Get-ChildItem -Path 'C:\Users\user4\*' -File -Include *.py,*.ps1)
Select-String -Path ($files | ForEach-Object { $_.FullName }) -Pattern 'probes_5sets|probes_L20|zeroshot_paper16|bank_run|smoke2|q8_trained_SMOKE|falsify_smoke|axes_patched|passC_smoke|passP_smoke|lengthctl|confounds_L|expLBactivationLAYER22|axes_all\.pt|scale_70b|paper16_70b' | Select-Object -First 40 | ForEach-Object { '{0}:{1}: {2}' -f $_.Filename, $_.LineNumber, $_.Line.Trim().Substring(0, [Math]::Min(180, $_.Line.Trim().Length)) }
