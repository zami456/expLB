# READ-ONLY. Select-String over a wildcard set of lab files (e.g. C:/Users/user4/expLB/logs/*); UTF-16/UTF-8 aware.
# Prints file:line: text (first MAX hits). Sent via -EncodedCommand; writes nothing on the lab PC.
# Usage: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F082-F109_sgrepm.ps1 "PATH=C:/Users/user4/expLB/logs/*" "PAT=regex" MAX=200
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Select-String -Path '__PATH__' -Pattern '__PAT__' | Select-Object -First __MAX__ | ForEach-Object { '{0}:{1}: {2}' -f $_.Filename, $_.LineNumber, ($_.Line.Substring(0, [Math]::Min(300, $_.Line.Length))) }
