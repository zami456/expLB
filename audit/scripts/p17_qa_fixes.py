"""P17 QA fixes, F1/F3/F4 of audit/p17/QA_REPORT.md. The lead writes these; they are INSERT-ONLY.

Each site is given as (file, original line, expected substring). The script checks the substring is
on that line, walks back to the start of the paragraph/bullet/table containing it, and inserts the
block there. Files are snapshotted to audit/p17/pre_qafix/ first. Line endings are preserved.
Re-running is a no-op: a site is skipped if its block text is already present in the file.
"""
import pathlib, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SNAP = ROOT / "audit/p17/pre_qafix"

F1 = ("> **CORRECTION 2026-09-24 (P17 QA F1).** \"165/9/2\" counts only the 176 draft claims that Phase 6A "
      "checked. All **237** draft (D-) claims in `audit/claims.csv` carry a verdict: 208 MATCHES HIS DATA, "
      "8 HOLDS, 18 PARTLY (12 PARTLY MATCHES + 2 PARTLY HOLDS + 4 annotated PARTLY), 2 UNRESOLVED, "
      "1 SUPERSEDED. **Source:** `audit/claims.csv` (ids D-*); `audit/p17/QA_REPORT.md` F1.")
F1B = ("- **Correction 2026-09-24 (P17 QA F1):** \"draft 176/176 at 165/9/2\" is the 176 Phase-6A rows only; "
       "all 237 draft claims: 208 MATCHES / 8 HOLDS / 18 PARTLY / 2 UNRESOLVED / 1 SUPERSEDED. "
       "Source: `audit/claims.csv` (ids D-*); `audit/p17/QA_REPORT.md` F1.")
F3B = ("- **Correction 2026-09-24 (P17 QA F3):** 128 entries, not 127 — F-128 was appended after this "
       "entry was drafted. Source: `kd/docs/FINDINGS.md` (`grep -c '^### F-'` = 128).")
PM = ("> **CORRECTION 2026-09-24 (DF-101, P17).** Excluding HP-KR, the ext_axis1 − ext_pc3 margin reaches "
      "**+0.0633 at L56**, so it is not within ±0.06 (at 4 dp), and it changes sign **6** times, not 5. "
      "**Source:** `src/probes/deception-detection/results/persona_ood_ext_L{20..79}/summary.csv` @ 16def2de; "
      "`kd/docs/FINDINGS.md` F-128; `audit/p17/QA_REPORT.md` check 3.")
TRAP = ("> **CORRECTION 2026-09-24 (DF-100, P17).** The bank-store \"datapca trap\" (`axes` wins 0 of 80) "
        "does not reproduce from the current files: the passA11 bank matrices have no `datapca` column, and "
        "the bank summary's `datapca` equals the root store's, which gives 31/80 (28/80 against raw). Whether "
        "an earlier version of the store read 0/80 is UNKNOWN. **Source:** `audit/p17/C_REPORT.md` (F-111 row); "
        "`audit/p17/QA_REPORT.md` check 3; `kd/docs/FINDINGS.md` F-128.")
REG = ("> **UPDATE 2026-09-24 (DF-105, P17).** **0.9597 is unverified:** the regex behind it is recorded "
       "nowhere. A \"does-not-affirm\" reconstruction gives 0.9413, which is still above the draft's 0.919 "
       "(D-022). 0.9125 is verified. **Source:** `audit/p17/correction_blocks_check.csv`; `audit/p17/C_REPORT.md`; "
       "`audit/p17/QA_REPORT.md`.")

# (file, line, expected substring, [blocks], bullet_form)
SITES = [
    ("CLAUDE.local.md", 22, "176/176: 165/9/2", [F1], False),
    ("CLAUDE.local.md", 58, "176/176 (165/9/2)", [F1], False),
    ("audit/DOC_FIXES.md", 954, "draft 176/176 at 165/9/2", [F1B], True),
    ("audit/DOC_FIXES.md", 1095, "127 entries", [F3B], True),
    ("audit/DOC_FIXES.md", 794, "0.9597", [REG], False),
    ("audit/phase6/APPLY_SHEET.md", 268, "five times", [PM, REG], False),   # table rows 268 and 278
    ("audit/phase6/APPLY_SHEET.md", 238, "0.9597", [REG], False),
    ("audit/HPKR_WORDING_CONTROLLED.md", 38, "0.9597", [REG], False),
    ("audit/HPKR_WORDING_CONTROLLED.md", 288, "0.9597", [REG], False),
    ("audit/CORRECTIONS.md", 614, "0.9597", [REG], False),
    ("audit/phase4/RESULTS.md", 151, "0.9597", [REG], False),
    ("audit/PROGRESS.md", 2142, "0.9597", [REG], False),
    ("audit/PROGRESS.md", 2330, "0.9597", [REG], False),
    ("audit/PROGRESS.md", 547, "0 of 80", [TRAP], False),
    ("audit/phase3/CONTROLS.md", 112, "0 of 80", [TRAP], False),
    ("audit/phase5/digest_story.md", 459, "0 of 80", [TRAP], False),
    ("audit/claims/S-29.md", 65, "In the bank store", [TRAP], False),
]


def is_start(lines, i):
    s = lines[i].strip()
    prev = lines[i - 1].strip() if i > 0 else ""
    if s.startswith("|"):
        return not prev.startswith("|")
    return prev == "" or s.startswith(("- ", "* ", "#", ">")) or (s[:3].rstrip(".").isdigit() and ". " in s[:5])


def main():
    SNAP.mkdir(parents=True, exist_ok=True)
    byfile = {}
    for s in SITES:
        byfile.setdefault(s[0], []).append(s)
    log = []
    for f, sites in byfile.items():
        p = ROOT / f
        raw = p.read_bytes()
        nl = "\r\n" if b"\r\n" in raw else "\n"
        text = raw.decode("utf-8")
        snap = SNAP / f.replace("/", "__")
        if not snap.exists():
            shutil.copy(p, snap)
        lines = text.split(nl)
        ins = []
        for (_, ln, sub, blocks, bullet) in sites:
            i = ln - 1
            assert sub in lines[i], f"{f}:{ln} does not contain {sub!r}: {lines[i][:80]!r}"
            if all(b in text for b in blocks):
                log.append(f"SKIP (present) {f}:{ln}"); continue
            j = i
            if not bullet:
                while j > 0 and not is_start(lines, j):
                    j -= 1
            new = [b for b in blocks if b not in text]
            if bullet:
                chunk = new
            else:
                chunk = ([""] if j > 0 and lines[j - 1].strip() != "" else []) + \
                        [x for b in new for x in (b, ">")][:-1] + [""]
            ins.append((j, chunk))
            log.append(f"{f}: site line {ln} -> inserted before original line {j + 1} ({len(new)} block(s))")
        for j, chunk in sorted(ins, key=lambda t: -t[0]):
            lines[j:j] = chunk
        p.write_bytes(nl.join(lines).encode("utf-8"))
    print("\n".join(log))


if __name__ == "__main__":
    main()
