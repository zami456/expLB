$root = "C:/Users/user4/expLB/run-crossgen"
$stores = @("kd11/banks/collab")
"store,pooling,layer,seed,bank,bank_name,k,direction,dataset,auroc"
foreach ($s in $stores) {
  $dir = Join-Path $root "$s/matrices"
  if (-not (Test-Path $dir)) { "MISSING,$s,,,,,,,," ; continue }
  Get-ChildItem $dir -Filter "*_s42.json" | Sort-Object Name | ForEach-Object {
    $d = Get-Content $_.FullName -Raw | ConvertFrom-Json
    $ds = $d.datasets
    foreach ($r in $d.zeroshot) {
      for ($i = 0; $i -lt $ds.Count; $i++) {
        "{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}" -f $s, $d.pooling, $d.layer, $d.seed, $d.bank, $d.bank_name, $d.k, $r.direction, $ds[$i], $r.auroc[$i]
      }
    }
  }
}
