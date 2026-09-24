#!/usr/bin/env python
"""M1 closing checks: head counts for the F-42 arithmetic, the gate-3 extraction log,
and the mtimes/provenance of the files F-107 read.  READ-ONLY."""
import datetime
import glob
import hashlib
import json
import re
from pathlib import Path

R = Path("C:/Users/user4/expLB")
HUB = Path.home() / ".cache" / "huggingface" / "hub"


def ts(p):
    return datetime.datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")


print("== F-42 arithmetic inputs: num_attention_heads from the cached configs ==")
for name in ("models--meta-llama--Llama-3.1-8B-Instruct",
             "models--meta-llama--Llama-3.2-3B-Instruct",
             "models--meta-llama--Llama-3.3-70B-Instruct"):
    hits = glob.glob(str(HUB / name / "snapshots" / "*" / "config.json"))
    if not hits:
        print(f"   {name}: config.json not cached")
        continue
    c = json.loads(Path(hits[0]).read_text(encoding="utf-8"))
    h = c.get("num_attention_heads") or c.get("text_config", {}).get("num_attention_heads")
    kv = c.get("num_key_value_heads")
    nl = c.get("num_hidden_layers")
    print(f"   {name.split('--')[-1]:28} heads {h}  kv_heads {kv}  layers {nl}  "
          f"hidden {c.get('hidden_size')}")
    for T, budget in ((2048, 0.75), (2048, 0.55), (2048, 0.42), (1024, 0.55)):
        b = h * T * T * 4
        cap = budget * (1024 ** 3)
        print(f"       T={T} 1 row = {b:,} B = {b / 1024**2:8.2f} MiB = {b / 1e6:8.2f} MB ; "
              f"budget {budget} GiB = {cap / 1024**2:7.1f} MiB -> rows admitted "
              f"{int(cap // b)}")

print("\n== gate-3 extraction log ==")
for cand in (R / "logs" / "gate3_coords.log",):
    if not cand.exists():
        print("   MISSING", cand)
        continue
    raw = cand.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
    txt = raw.decode(enc, "replace")
    lines = [l.rstrip() for l in txt.splitlines() if l.strip()]
    print(f"   {cand}  {len(raw):,} B  enc={enc}  lines={len(lines)}  mtime {ts(cand)}")
    keep = [l for l in lines if not re.search(
        r"\[K|RECEIVED|QUEUED|DISPATCHED|RUNNING|COMPLETED|HTTP Request|%\|", l)]
    for l in keep[:60]:
        print("     ", l[:190])

print("\n== provenance of the files F-107 read ==")
for p in [R / "run-coords/sarcasm/sarcasm_natural/items.pt",
          R / "run-coords/sarcasm/sarcasm_natural/chunk_0000.pt",
          R / "run-coords/sarcasm/sarcasm_natural/chunk_0001.pt",
          R / "run-coords/sarcasm/sarcasm_natural/chunk_0002.pt",
          R / "run-coords/sarcasm/sarcasm_natural/rows.jsonl",
          R / "run-kd/axes_vectors_s23.pt",
          R / "run-kd/axes_collab_s23.pt",
          R / "cap/run-sarcasm-natural/rows.jsonl",
          R / "cap/run-sarcasm-natural/baselines.json",
          R / "cap/scripts/build_sarcasm_natural.py"]:
    if not p.exists():
        print(f"   MISSING {p}")
        continue
    h = hashlib.md5(p.read_bytes()).hexdigest() if p.stat().st_size < 200_000_000 else "SKIP_BIG"
    print(f"   {str(p).replace(str(R), ''):58} {p.stat().st_size:>12,} B  {ts(p)}  {h}")

print("\n== rows.jsonl of the extracted sarcasm store vs the prepared set ==")
a = [json.loads(l) for l in (R / "run-coords/sarcasm/sarcasm_natural/rows.jsonl").open(encoding="utf-8")]
b = [json.loads(l) for l in (R / "cap/run-sarcasm-natural/rows.jsonl").open(encoding="utf-8")]
print(f"   extracted {len(a)} rows, keys {sorted(a[0])}")
print(f"   prepared  {len(b)} rows")
print(f"   row_id sets identical: {set(x['row_id'] for x in a) == set(x['row_id'] for x in b)}")
lab_a = {x['row_id']: x['label'] for x in a}
lab_b = {x['row_id']: x['label'] for x in b}
print(f"   labels identical: {lab_a == lab_b}")
sl = [x.get("span_len") for x in a if x.get("span_len") is not None]
if sl:
    import numpy as np
    s = np.array(sl)
    print(f"   span_len n={len(s)} min {s.min()} p50 {np.percentile(s,50):.0f} "
          f"max {s.max()} zero-length {int((s==0).sum())}")

print("\n== SARCASM_DESIGN.md step 3 wording (the pre-registered rule) ==")
d = R / "cap" / "docs" / "SARCASM_DESIGN.md"
if d.exists():
    t = d.read_text(encoding="utf-8", errors="replace")
    i = t.lower().find("step 3")
    print(f"   {d} mtime {ts(d)}  len {len(t)}")
    print(t[max(0, i - 200): i + 1800] if i >= 0 else t[:1500])
else:
    print("   MISSING", d)
