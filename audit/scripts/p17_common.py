"""P17 shared helpers: every read of another branch goes through a PINNED SHA, never a moving ref.

Pins recorded in audit/BRANCH_PINS.md (2026-09-24, after one `git fetch`).
Helper bodies follow p16_level1.py:15-24 / p16_level2.py:22-40, with the SHA as an argument.
"""
import csv, io, subprocess, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PIN_OLD = "ea1ce52b874e227b2536e1c0f4b31551ff8e08a7"       # last origin/paper commit the Phase-6 audit claims to have read
PIN_PAPER = "16def2deec34aeee532511df116adb7de5522ca7"   # origin/paper after fetch 2026-09-24
PIN_PCOOD = "45b16d906da0e3da13b1abb34898e69311d688cf"   # origin/feat/persona-pc-ood (ancestor of paper)
PIN_ASIX = "e917ecd3f95f8ec16c0ce3816e9878c0bacd4c7e"    # asix HEAD (+ uncommitted WT, md5s in BRANCH_PINS.md)
PIN_P2AD = "2ad758fb38c28dd2504ac358f7169148882b922f"; PIN_PD1B = "d1b8815402e02f28c5dbbb95ce4eb079a7ecefba"            # intermediate commits (provenance check)
BASE = "src/probes/deception-detection"
RES = BASE + "/results"


def show(sha, path):
    r = subprocess.run(["git", "show", f"{sha}:{path}"], cwd=ROOT, capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return r.stdout if r.returncode == 0 else None


def tab(sha, path):
    t = show(sha, path)
    return list(csv.DictReader(io.StringIO(t))) if t and t.strip() else None


def fl(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None
