"""Writes audit/phase1/sessions/findings_rerun_origin.md (transcript-mining deliverable 3).

Content is DOC-ONLY evidence quoted from the owner's Claude session transcripts.
Re-run: python audit/scripts/p1_sessf_write_rerun.py
"""
import io
import os

DOC = r'''# Who/what triggered the two 2026-09-22 re-run events (RF-13, and the 00:52-01:33 sweep)

**Status: DOC-ONLY.** Everything below is quoted from the owner's Claude session transcripts.
Transcripts are model and user text, not verified data. The file mtimes, logs and `.status`
files Phase 1 already measured are the verified side; this file only explains them.

Source transcript:
`C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/1ac8bb19-f55a-4a95-91bb-fdaee3d53699.jsonl`
(session `1ac8bb19`, 2026-09-21T17:12Z - 2026-09-22T12:07Z). Transcript timestamps are **UTC**;
the audit's lab-PC times are **local = UTC+06**.

---

## Event 1 - the 70B k=3 sweep, 2026-09-22 00:52-01:33 local (2026-09-21 18:52-19:33Z)

**Trigger: the owner, through this Claude session. Nothing automatic.**

- `L831` `2026-09-21T18:40:19Z` (00:40 local) - owner message, verbatim:
  `all 3, step by step, first tell me can i sahre my collaborator debriefing?does my claim stand`
- `L995` `2026-09-21T18:51:07Z` - the assistant's own description of what it was about to do:
  `The 70B rung was run by hand (there's no run-scale/70b), against passB with axes_vectors_s3.pt.
  Launching the sweep with the same settings, mean pooling - which is what F-96 measured.`
- `L996` `2026-09-21T18:51:15Z` - the launching command (`tool_use:Bash`, `run_in_background=true`,
  written to `task2.sh` in the session scratchpad and executed through `psrun.sh`). Verbatim core:

```
$DS = @("instructed-deception","mask","convincing-game","ai_liar","roleplaying")
$md = "$r\run-crossgen\scale_70b\matrices"
for ($try=1; $try -le 5; $try++) {
  $have = @(Get-ChildItem $md -Filter "mean_L*_s42.json" -EA SilentlyContinue).Count
  if ($have -ge 21) { Write-Output "all 21 present"; break }
  $a = @("$r\src\assistant_axis_ndif\27_crossgen.py","--run_dir","run-coords\passB",
    "--out_root","run-crossgen\scale_70b","--bank_name","paper16_70b",
    "--bank","$r\run-kd\axes_vectors_s3.pt","--datasets") + $DS + @(
    "--k","3","--poolings","mean","--stage","all")
  & "$r\.venv-analysis\Scripts\python.exe" $a *> "$env:TEMP\scale70b_$try.log"
}
```

**This is the primary-source confirmation of RF-16.** The command passes
`--bank_name paper16_70b` while pointing `--bank` at `run-kd/axes_vectors_s3.pt` - the
36-persona **kd** bank, not `run-paper16/axes_vectors_s3.pt`. The two files share a basename;
the assistant read "run by hand ... with `axes_vectors_s3.pt`" off the existing L40 cell's
metadata and resolved that basename to the wrong folder. The mislabelled matrices are the ones
F-100 / F-101 / F-102 average over. (The explanation is DOC-ONLY; the bank mix itself is
VERIFIED independently by the audit's zero-shot fingerprint, RF-16.)

- The sweep ran in the background as task `baro2wq0l`, retrying past out-of-memory failures
  (four foreign `run_benchmark_*` jobs from another lab account had free RAM down to 0.2-0.6 GB),
  reached 19/21 layers when F-100 was first written, then 21/21:
  - `L1019` `19:02:01Z` - `This closes F-96's open gap. Writing it up as F-100.`
  - `L1038` `19:03:24Z` - `Complete - all 21 landed, and the numbers barely moved. Amending
    F-100 in place, matching the F-94 precedent.`
- `L1075` `2026-09-21T19:27:27Z` (01:27 local) - owner: `run the two optionals`. That started
  (a) a `last`-pooling background sweep and (b) a manual `Start-ScheduledTask KD-S23` re-trigger,
  whose status file ends `2026-09-22 01:33:21  DONE` - the 01:33 end of the audit's window.

**Conclusion (INFERRED, DOC-ONLY):** event 1 is fully accounted for as owner-requested work
carried out by Claude session `1ac8bb19`. It predates the audit (18:44 local on 09-22), which is
consistent with the owner's statement that she ran nothing on the lab PC after the audit began.

---

## Event 2 - the wave, 2026-09-22 15:40-16:26 local (09:40-10:26Z) = RF-13

**Trigger: the lab PC REBOOTED at 15:34:30 local, and ~24 project Scheduled Tasks carry
`MSFT_TaskBootTrigger` with PT20M / PT25M / PT30M delays. No person and no Claude session
started them.**

- `L2302` `2026-09-22T09:18:16Z` - the box went unreachable: `SSH x 3 Connection timed out`,
  `ICMP ping x 2 100% loss`, `Tailscale peer active; relay "blr" - tx 1248 rx 0`.
- `L2344` `2026-09-22T09:46:31Z` - `tool_res`, verbatim:

```
NOW 2026-09-22 16:01:19  |  booted 09/22/2026 15:34:30
KD-S23             State=Running rc=267009 Next=none
   triggers: MSFT_TaskBootTrigger delay=PT25M
Scale-3b           State=Ready   rc=0 Next=none
   triggers: MSFT_TaskBootTrigger delay=PT30M
Scale-8b           State=Ready   rc=0 Next=none
   triggers: MSFT_TaskBootTrigger delay=PT30M
KD-Bank-Crossgen   State=Running rc=267009 Next=none
   triggers: MSFT_TaskBootTrigger delay=PT20M
```

  Produced by the `tool_use:Bash` at `L2343`: a `psrun.sh` block calling
  `(Get-CimInstance Win32_OperatingSystem).LastBootUpTime` and `Get-ScheduledTask` for the four names.

- `L2351` `2026-09-22T10:01:53Z` - assistant, verbatim:
  `Correction 1: these tasks were never trigger-less. All four carry MSFT_TaskBootTrigger:`
  `KD-Bank-Crossgen PT20M -> 15:54:30 fired; KD-S23 PT25M -> 15:59:30 fired;
  Scale-3b / Scale-8b PT30M -> 16:04:30 - imminent.`
  `Correction 2: I misread Next=none. I took it to mean "cannot fire on its own" and said so
  across roughly twenty checks. It only means no time-scheduled next run - a boot trigger
  doesn't populate NextRunTime.`

- `L2349` `2026-09-22T10:01:42Z` - `tool_res` carrying the status lines the audit later found:

```
=== kd_bank_crossgen.status  (27 lines, mt 16:01:30)
2026-09-22 15:57:54  STEP peak exit 0
2026-09-22 15:57:57  STEP l20 layers=20 start
2026-09-22 15:59:26  STEP l20 exit 0
2026-09-22 15:59:30  STEP rest layers=0,4,8,12,16,22,24,28,32,36,48,52,56,60,64,68,72,76 start
2026-09-22 16:01:21  STEP rest exit 0
2026-09-22 16:01:30  DONE
=== kd_s23.status  (34 lines, mt 16:00:25)
2026-09-22 01:33:21  DONE
2026-09-22 16:00:01  WAITING for KD-Chain to finish (NDIF must be free before extracting)
2026-09-22 16:00:01  KD-Chain done; extracting vectors_s23
2026-09-22 16:00:22  vectors_s23 exit 0: 36 vectors (expect 36)
2026-09-22 16:00:25  axes_s23 exit 0
2026-09-22 16:00:25  crossgen s23 peak layers=40,44
```

  The same output lists **eight python processes** - "four parent/child pairs: the two chain
  crossgens, plus `30c_cluster_perm` and another `27_crossgen`, so other work resumed on boot too."

- `L2365` `2026-09-22T10:16:30Z` - the outcome the assistant reported, verbatim:
  `Nothing to append. Every matrix was resumed, not recomputed:`
  `crossgen s23 peak exit 0: 4/4 matrices present (0 rebuilt, 4 resumed)`
  `crossgen s23 l20  exit 0: 2/2 matrices present (0 rebuilt, 2 resumed)`
  `No new measurement was produced, so FINDINGS stays at 5,136 lines.`
  Scale-3b and Scale-8b both reported `SKIP already DONE; -Force to re-run`.

- `L2757` `2026-09-22T11:31:08Z` - `a reboot fires ~24 chains over 50 minutes - that's what
  happened at 15:34.`

**Caveat for Phase 2.** "Every matrix was resumed, not recomputed" covers only the two chains
the session inspected (KD-Bank-Crossgen, KD-S23). RF-13 counts ~58,000 rewritten lab files, and
RF-14 shows the `^DONE` guard bug makes the chain *scripts* re-run and truncate their `.status`
files even when the per-cell results are skipped. Phase 1's own keyed comparison - git is an
exact subset of the lab copy with 0 differing metric values, only BH `q` moved in
`significance.json` - is the evidence. This quote is corroboration, not proof.

**Why the box rebooted was never established.** The session diagnosed *that* it rebooted
(`LastBootUpTime`), never *why*. No Windows event-log query (ID 1074 / 6008) appears in any
transcript. The lab PC is multi-user and foreign `run_benchmark_*` jobs from another account had
been exhausting its RAM the night before. **NOT-FOUND: who or what rebooted the box.**

---

## Third fact: 24 chains were disabled ~73 minutes before the audit began

`L2763` `2026-09-22T11:31:32Z` (17:31 local), on the owner's instruction
`stop any schedule task that requires ur attention` (`L2743`, 17:30 local). Verbatim:

```
Done - 24 boot-triggered chains disabled, including all four you'd been asking me to poll.
Disabled: every Cap-*, KD-*, Crossgen-*, Scale-*, Premise-*, plus Sarc-Pilot, Falsify-MeanDir,
HPKR-Topup.
Kept on purpose - these three protect your data: KD-Sync-Periodic, Crossgen-Sync-Periodic,
Coords-Supervisor.
```

The re-enable snippet it printed names all 24:
`Cap-Chain, Cap-Controls, Cap-Ctrl-Layers, Cap-Ctrl-XG, Cap-Layers, Cap-Seeds, Crossgen-Chain,
Crossgen-Final-Report, Falsify-MeanDir, HPKR-Topup, KD-Bank-Crossgen, KD-Chain, KD-Crossgen,
KD-Extract, KD-Knowledge, KD-PassAC, KD-S23, KD-S23-AllLayers, Premise-Extras, Premise-Layers,
Sarc-Pilot, Scale-3b, Scale-8b, Scale-Coords`

This explains Phase 0's observation that 5 tasks still fire every 15-30 min while "all other
project tasks" are Disabled: the disabling was an action taken by the previous Claude session at
17:31 local on 2026-09-22, not a long-standing state. The lab PC is therefore currently much
quieter than it was during the week the results were produced - and a Phase 2 re-run that
re-enables anything would restore the reboot cascade.

---

## Fourth fact: the Drive-sync outage behind part of RF-10

- The lab PC's rclone OAuth token for `tamara:` stopped working at **12:18 local on 2026-09-22**;
  every sync failed for five hours. That window contains the two CRITICAL rclone lines Phase 1
  found (`tamara:expLB/coords/passC/...` at 15:47, `tamara:expLB/scale: couldn't find root
  directory ID` at 16:20).
- `L2703`-`L2727` `2026-09-22T11:24-11:26Z` (17:24-17:26 local): the owner ran
  `rclone config reconnect tamara:` in her own PowerShell window; the refreshed token was then
  copied to the lab PC - `1 token line replaced, backup at rclone.conf.bak-2026-09-22`.
- `L2735` `2026-09-22T11:28:39Z` - verbatim:

```
17:27:11  periodic run-kd      exit 0  (184 files)
17:27:53  periodic crossgen11  exit 0  (68,666 files)
17:27:53  periodic DONE        exit 0
17:28:20  manual <whole tree>  EXIT 0
```

  `First exit 0 since 12:18 - five hours of failures ended.`

So the backlog was uploaded at 17:27-17:28 local, ~90 minutes before the audit's census (20:56).
Folders the census still found missing from Drive are therefore missing for a different reason -
most likely the sync scripts' `--include "*.pt" --include "*.json"` filters and their per-folder
scope. **A Phase 2 check, not a conclusion.**

Note also: that token transplant was a **write to the lab PC by a previous Claude session**
(`rclone.conf` plus `rclone.conf.bak-2026-09-22`), i.e. lab-PC state changed 78 minutes before
the audit's baseline.

---

## Re-runnable commands

```
FLAT=<scratchpad>/flat            # built by audit/scripts/p1_sessf_flatten.py
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_find.py "$FLAT" "MSFT_TaskBootTrigger" --sess 1ac8bb19
PYTHONIOENCODING=utf-8 python audit/scripts/p1_sessf_find.py "$FLAT" "LastBootUpTime" --sess 1ac8bb19
awk -F'\t' '$2==996 {print $4"#"$5": "$6}' "$FLAT/flat_1ac8bb19.tsv"    # the task2.sh launch (RF-16)
awk -F'\t' '$2==2763{print $4"#"$5": "$6}' "$FLAT/flat_1ac8bb19.tsv"    # the 24 disabled tasks
awk -F'\t' '$3>="2026-09-22T09:15" && $3<="2026-09-22T10:40"' "$FLAT/flat_1ac8bb19.tsv"
```
'''

out = os.path.join("audit", "phase1", "sessions", "findings_rerun_origin.md")
with io.open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write(DOC)
print("wrote", out, len(DOC), "chars")
