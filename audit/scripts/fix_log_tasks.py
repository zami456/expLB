"""Log the 2026-09-23 owner-authorised task changes into audit/LABPC_WRITES.md and audit/PROGRESS.md."""
import datetime

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
open("audit/LABPC_WRITES.md", "a", encoding="utf-8").write(
    f"| {now} | SCHEDULED TASKS DISABLED (owner-authorised) | lab PC tasks PassAC-11, PassC-PerToken, Night-Chain, "
    "Coords-Supervisor | their work is complete (passAC11.status DONE; run-coords passA/B/C .done present, so "
    "run_passC / night_chain / coords_supervisor exit early and only synced). Disabled so a reboot cannot re-run them "
    "(RF-13). Coords-Supervisor was Running (an rclone check) and was stopped first. Re-enable with "
    "Enable-ScheduledTask -TaskName '<name>'. Evidence audit/fix/task_completion.txt; result "
    "audit/fix/task_disable_result.txt | 4 tasks |\n")

block = """## Owner items closed 2026-09-23 (after the fix round)
- **F-110 appended to kd/docs/FINDINGS.md** (owner said yes): the RF-16 correction, in the ledger's style, marked as written by the audit, with the like-for-like table, sources and caveats. `git diff --ignore-cr-at-eol` = 53 added lines and nothing else; the full-file churn in a plain `git diff` is a pre-existing CRLF-vs-LF checkout artefact (the HEAD blob is LF, the working file is CRLF) and disappears on commit.
- **The 8 script fixes stay uncommitted** — the owner will commit them.
- **Completed lab tasks disabled** (owner: "are they already completed? if yes then stop them"). Verified complete first, then disabled: PassAC-11 (status DONE 16:05:43), and PassC-PerToken / Night-Chain / Coords-Supervisor (run-coords passA/B/C `.done` present, so they exit early and only synced).
  - **Kept enabled on purpose:** KD-Sync-Periodic and Crossgen-Sync-Periodic — pure `rclone copy` to Drive; they cannot re-run an experiment, and Crossgen-Sync-Periodic is what backs up the new run-crossgen/scale_70b_paper16fp32 folder.
  - Every other project task is now Disabled or has no trigger (manual only).
  - **Flagged, not touched:** `Thesis_Train` is Ready with a boot trigger and runs run_train.ps1 under D:/user4fina/code (last run 2026-08-04). It belongs to the owner's older thesis work, outside this project — her call.
  - Unrelated: two python processes on the shared lab PC belong to another team (under C:/Users/user5, SYCON-Bench), not to this project.

## Phase 2 watch list (collected during Phase 1)"""
p = "audit/PROGRESS.md"
s = open(p, encoding="utf-8").read()
assert "## Phase 2 watch list (collected during Phase 1)" in s
open(p, "w", encoding="utf-8").write(s.replace("## Phase 2 watch list (collected during Phase 1)", block, 1))
print("recorded")
