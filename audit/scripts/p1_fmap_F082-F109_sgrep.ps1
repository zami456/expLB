# READ-ONLY. Select-String over one lab file (auto-detects UTF-16/UTF-8 BOM, unlike p1_peek.py grep).
# Sent via -EncodedCommand; writes nothing on the lab PC.
# Usage: python audit/scripts/labpc_run.py audit/scripts/p1_fmap_F082-F109_sgrep.ps1 PATH=C:/Users/user4/expLB/logs/x.log "PAT=regex" MAX=200
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
Select-String -LiteralPath '__PATH__' -Pattern '__PAT__' | Select-Object -First __MAX__ | ForEach-Object { '{0}: {1}' -f $_.LineNumber, ($_.Line.Substring(0, [Math]::Min(400, $_.Line.Length))) }
