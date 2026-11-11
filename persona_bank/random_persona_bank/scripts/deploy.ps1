<#
  Deploy one run folder per model: random_persona_bank\results\<Model>\code = xmodel/code's pipeline + this bank.

      code\src, code\assistant_axis   from xmodel\code (the pipeline every xmodel run used; kd\ left out -- unused here)
      code\rpb                        from random_persona_bank\bank, checked against bank\MANIFEST.md5
      code\run_chain.ps1, sync_drive.ps1   from random_persona_bank\scripts (refreshed on every deploy)

  The AA judge prompts say "(an AI assistant, Gemma, an LLM, etc.)" -- AA wrote them for Gemma. As xmodel does for the
  kd prompts, the deployed copy names the model actually judged; asserted: 35 files x 3 occurrences renamed and no
  "Gemma" left (Gemma's own run keeps them, asserted present). The bank in random_persona_bank\bank is never edited.

  Laptop only. No NDIF, no task registered: register_queue.ps1 does that.

      powershell -ExecutionPolicy Bypass -File random_persona_bank\scripts\deploy.ps1 [-Tags qwen,gemma3]
#>
param([string[]]$Tags = @("llama33", "qwen", "mistral2501", "mistral31", "gemma3"))
$ErrorActionPreference = "Stop"
$known = @{ "llama33" = @("Llama-3.3-70B-Instruct", "Llama"); "qwen" = @("Qwen2.5-72B-Instruct", "Qwen")
            "mistral2501" = @("Mistral-Small-24B-Instruct-2501", "Mistral")
            "mistral31" = @("Mistral-Small-3.1-24B-Instruct-2503", "Mistral"); "gemma3" = @("gemma-3-27b-it", "Gemma") }
$Base    = Split-Path $PSScriptRoot -Parent              # random_persona_bank
$Repo    = Split-Path $Base -Parent
$Results = Join-Path $Base "results"
$Bank    = Join-Path $Base "bank"
$Phrase  = "an AI assistant, Gemma, an LLM"

foreach ($tag in $Tags) {
    if (-not $known.ContainsKey($tag)) { throw "unknown tag $tag" }
    $Folder, $Short = $known[$tag]
    $Code = Join-Path (Join-Path $Results $Folder) "code"
    if (-not (Test-Path $Code)) {
        New-Item -ItemType Directory -Force -Path $Code | Out-Null
        robocopy (Join-Path $Repo "xmodel\code") $Code /E /XD __pycache__ kd /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy xmodel\code failed ($LASTEXITCODE)" }
        robocopy $Bank (Join-Path $Code "rpb") /E /XD __pycache__ /XF build_bank.py /NFL /NDL /NJH /NJS | Out-Null
        if ($LASTEXITCODE -ge 8) { throw "robocopy bank failed ($LASTEXITCODE)" }
        # the copy is the bank, byte for byte, before the rename
        $bad = 0
        foreach ($line in Get-Content (Join-Path $Bank "MANIFEST.md5")) {
            $want, $rel = $line -split "  ", 2
            $got = (Get-FileHash -Algorithm MD5 (Join-Path (Join-Path $Code "rpb") $rel)).Hash.ToLower()
            if ($got -ne $want) { $bad++; Write-Output "  md5 differs: $rel" }
        }
        if ($bad) { throw "$bad deployed bank files differ from MANIFEST.md5" }
        $instr = Join-Path $Code "rpb\rpb_instructions"
        $n = 0; $occ = 0
        foreach ($f in Get-ChildItem $instr -Filter *.json) {
            $t = [IO.File]::ReadAllText($f.FullName)
            $k = ([regex]::Matches($t, [regex]::Escape($Phrase))).Count
            if ($k) { $n++; $occ += $k }
            if ($k -and $Short -ne "Gemma") {
                [IO.File]::WriteAllText($f.FullName, $t.Replace($Phrase, "an AI assistant, $Short, an LLM"))
            }
        }
        if ($n -ne 35 -or $occ -ne 105) { throw "judge prompts: $n files / $occ occurrences carry '$Phrase' (want 35 / 105)" }
        if ($Short -ne "Gemma") {
            $left = @(Select-String -Path (Join-Path $instr "*.json") -Pattern "Gemma").Count
            if ($left) { throw "judge prompt rename: $left Gemma mentions left" }
            Write-Output "$Folder : deployed; judge prompts renamed in 35 files (105 occurrences) to $Short"
        } else {
            Write-Output "$Folder : deployed; judge prompts already name Gemma (35 files, 105 occurrences)"
        }
    } else {
        Write-Output "$Folder : code already deployed (bank and judge prompts left as they are)"
    }
    Copy-Item (Join-Path $PSScriptRoot "run_chain.ps1"), (Join-Path $PSScriptRoot "sync_drive.ps1") $Code -Force
}
