"""The gates on kd_questions_shared_v2.jsonl. Exit 1 on any failure; prints the file's SHA-256.

Checks run against the 30 originals and the file itself only, never against any test set.

    python check_questions.py [path]
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
ORIGINALS = Path(r"E:\aletheia\banks\Qwen3.5-27B\code\kd\questions\kd_questions_shared.jsonl")
KD_ROWS = Path(__file__).resolve().parents[3] / "persona_bank" / "kd" / "run-kd" / "responses" / "default.jsonl"  # stored text of the originals
AUX = ("is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would", "will", "have",
       "has", "am", "shall", "may", "might", "must")
MAX_WORDS = {"bare": 60, "yesno": 60, "context": 90, "twoturn": 110}
MONO_RANGE = (15, 50)
JACCARD_MAX = 0.5


def words(s: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9']+", s.lower())


def norm(s: str) -> str:
    return " ".join(words(s))


def ngrams(ws: list[str], n: int) -> set[tuple]:
    return {tuple(ws[i:i + n]) for i in range(len(ws) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def main(path: Path) -> int:
    fails: list[str] = []
    raw_lines = path.read_text(encoding="utf-8").split("\n")
    if raw_lines[-1] == "":
        raw_lines = raw_lines[:-1]
    if any(not l.strip() for l in raw_lines):
        fails.append("blank line present")
    rows = [json.loads(l) for l in raw_lines if l.strip()]

    # 1 ids and schema
    if len(rows) != 150:
        fails.append(f"{len(rows)} rows, expected 150")
    if [r.get("id") for r in rows] != list(range(len(rows))):
        fails.append("ids are not the line order 0..149")
    for r in rows:
        if not isinstance(r.get("question"), str) or not r["question"].strip():
            fails.append(f"row {r.get('id')}: empty question")
        if not r["question"].rstrip().endswith("?"):
            fails.append(f"row {r.get('id')}: does not end with '?'")

    # 2 originals byte-identical, in order, and equal to the stored rollout text
    orig = [json.loads(l) for l in ORIGINALS.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, o in enumerate(orig):
        if rows[i]["question"] != o["question"]:
            fails.append(f"row {i}: original text differs from {ORIGINALS.name}")
    if KD_ROWS.exists():
        stored = {}
        for l in KD_ROWS.read_text(encoding="utf-8").splitlines():
            if l.strip():
                d = json.loads(l); stored[int(d["question_index"])] = d["question"]
        for i in range(30):
            if i in stored and stored[i] != rows[i]["question"]:
                fails.append(f"row {i}: differs from the stored rollout text in {KD_ROWS}")

    # 3 duplicates after normalisation
    seen = Counter(norm(r["question"]) for r in rows)
    for k, v in seen.items():
        if v > 1:
            fails.append(f"duplicate question: {k[:60]}...")

    # 4 lengths and 5 sets / grid margins
    sets = Counter(r["set"] for r in rows)
    if sets != {"orig": 30, "mono": 30, "div": 90}:
        fails.append(f"set counts {dict(sets)}")
    slots = json.loads((HERE / "grid_slots.json").read_text(encoding="utf-8"))
    by_slot = {s["slot"]: s for s in slots}
    for r in rows:
        n = len(words(r["question"]))
        if r["set"] == "mono":
            if not (MONO_RANGE[0] <= n <= MONO_RANGE[1]):
                fails.append(f"row {r['id']}: mono length {n} outside {MONO_RANGE}")
            g = r["grid"]
            if not (g and g["deceiver"] == "user" and g["domain"] == "workplace" and g["format"] == "bare"):
                fails.append(f"row {r['id']}: mono grid {g}")
        elif r["set"] == "div":
            g = r["grid"]; s = by_slot.get(r["slot"])
            if s is None or (g["imt"], g["deceiver"], g["domain"], g["format"]) != (s["imt"], s["deceiver"], s["domain"], s["format"]):
                fails.append(f"row {r['id']}: grid does not match slot {r['slot']}")
            if n > MAX_WORDS[g["format"]]:
                fails.append(f"row {r['id']}: {g['format']} length {n} > {MAX_WORDS[g['format']]}")
    div = [r for r in rows if r["set"] == "div"]
    margins = {k: Counter(r["grid"][k] for r in div) for k in ("imt", "deceiver", "domain", "format")}
    want = {k: Counter(s[k] for s in slots) for k in ("imt", "deceiver", "domain", "format")}
    if margins != want:
        fails.append(f"grid margins differ from grid_slots.json: {margins}")
    mono_types = Counter(r["grid"]["imt"] for r in rows if r["set"] == "mono")
    if any(v != 5 for v in mono_types.values()) or len(mono_types) != 6:
        fails.append(f"mono types {dict(mono_types)}")

    # 6 n-gram overlap among new items and against the originals
    new = [r for r in rows if r["set"] != "orig"]
    g4 = {r["id"]: ngrams(words(r["question"]), 4) for r in rows}
    g8_orig = {r["id"]: ngrams(words(r["question"]), 8) for r in rows if r["set"] == "orig"}
    worst = 0.0
    for a, b in combinations(new, 2):
        j = jaccard(g4[a["id"]], g4[b["id"]]); worst = max(worst, j)
        if j > JACCARD_MAX:
            fails.append(f"rows {a['id']} and {b['id']}: 4-gram Jaccard {j:.2f}")
    for a in new:
        w8 = ngrams(words(a["question"]), 8)
        for o in rows[:30]:
            j = jaccard(g4[a["id"]], g4[o["id"]]); worst = max(worst, j)
            if j > JACCARD_MAX:
                fails.append(f"rows {a['id']} and original {o['id']}: 4-gram Jaccard {j:.2f}")
            if w8 & g8_orig[o["id"]]:
                fails.append(f"row {a['id']} contains an 8-gram of original {o['id']}")

    # 7 format markers
    for r in div:
        q = r["question"].strip(); f = r["grid"]["format"]; ws = words(q)
        if f == "yesno" and not (ws[0] in AUX or "yes or no" in q.lower()):
            fails.append(f"row {r['id']}: yesno item does not start with an auxiliary or contain 'yes or no'")
        if f == "context":
            body = q[: q.rfind("?")]
            n_sent = len([s for s in re.split(r"(?<=[.!?])\s+", body) if s.strip()])
            if n_sent < 3:  # at least two sentences of context plus the question
                fails.append(f"row {r['id']}: context item has {n_sent} sentences")
        if f == "twoturn":
            if not re.search(r'Earlier I asked: ".+?" You said: ".+?"', q):
                fails.append(f"row {r['id']}: twoturn item lacks the fixed markers")

    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"rows {len(rows)}; max 4-gram Jaccard {worst:.2f}; sha256 {sha}")
    if fails:
        print("FAILED:")
        for f in fails:
            print("  " + f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "kd_questions_shared_v2.jsonl"
    sys.exit(main(p))
