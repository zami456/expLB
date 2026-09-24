"""Owner-authorised fix (2026-09-22) of two script bugs, applied byte-exactly (line endings preserved).

RF-14: in 7 chain scripts the skip guard `Select-String -Pattern '^DONE'` can never match, because Note() writes
       "<yyyy-MM-dd HH:mm:ss>  DONE". New pattern matches exactly that final line (tested read-only on the lab:
       audit/fix/guard_pattern_test.txt).
RF-6:  run_cap_ctrl_layers.ps1 lines 31-32 contain a literal backspace byte (0x08) where "\b" of "\banks" was meant.

Usage: python audit/scripts/fix_guard_bug.py <base_dir> <originals_dir>
  Patches the files under <base_dir> (a repo working tree, or a local copy of the lab tree) and saves the untouched
  originals under <originals_dir>. Refuses to write if a pattern is not found exactly the expected number of times.
"""
import hashlib, os, sys

OLD_GUARD = b"-Pattern '^DONE'"
NEW_GUARD = b"-Pattern '^\\d{4}-\\d{2}-\\d{2} \\d{2}:\\d{2}:\\d{2}\\s+DONE\\s*$'"
GUARD_FILES = [
    "cap/scripts/run_cap_controls.ps1", "cap/scripts/run_cap_layers.ps1", "cap/scripts/run_sarc_pilot.ps1",
    "kd/scripts/run_passAC_11.ps1", "src/assistant_axis_ndif/run_falsify.ps1",
    "src/assistant_axis_ndif/run_premise_extras.ps1", "src/assistant_axis_ndif/run_premise_layers.ps1",
]
BS_FILE = "src/assistant_axis_ndif/run_cap_ctrl_layers.ps1"
OLD_BS = b"kd11\x08anks"
NEW_BS = b"kd11\\banks"

base, orig = sys.argv[1], sys.argv[2]
plan = [(f, OLD_GUARD, NEW_GUARD, 1) for f in GUARD_FILES] + [(BS_FILE, OLD_BS, NEW_BS, 2)]
patched = []
for rel, old, new, expect in plan:
    p = os.path.join(base, rel)
    data = open(p, "rb").read()
    n = data.count(old)
    if n != expect:
        sys.exit(f"ABORT: {rel}: pattern found {n} times, expected {expect}; nothing written")
    patched.append((rel, data, data.replace(old, new)))
for rel, data, new_data in patched:          # all checks passed: save originals, then write
    o = os.path.join(orig, rel)
    os.makedirs(os.path.dirname(o), exist_ok=True)
    open(o, "wb").write(data)
    open(os.path.join(base, rel), "wb").write(new_data)
    print(f"{rel}: md5 {hashlib.md5(data).hexdigest()} -> {hashlib.md5(new_data).hexdigest()}  ({len(data)} -> {len(new_data)} bytes)")
