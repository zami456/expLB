"""W5B-ledger, lab pass E - READ-ONLY.  Last loose ends.
F-64: run-cap vector folders (2d gate: 12 each for s23 and all, predicted 13) + cap_chain.status.
F-55: every row of the no-persona probe file really has system_prompt = null.
"""
import json, os, glob

R = "C:/Users/user4/expLB"


def readtext(p):
    b = open(p, "rb").read()
    return b.decode("utf-16", "replace") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8", "replace")


print("== F-64  run-cap contents ==")
for d in sorted(glob.glob(f"{R}/run-cap/*")):
    if os.path.isdir(d):
        fs = os.listdir(d)
        print(f"  {os.path.basename(d):18} {len(fs)} entries  e.g. {sorted(fs)[:3]}")
    else:
        print(f"  {os.path.basename(d):18} FILE {os.path.getsize(d)} B")

p = f"{R}/logs/cap_chain.status"
print(f"\n== F-64  {p} ({os.path.getsize(p)} B, mtime "
      f"{__import__('datetime').datetime.fromtimestamp(os.path.getmtime(p)):%Y-%m-%d %H:%M}) ==")
for l in readtext(p).splitlines():
    print("  |", l[:190])

print("\n== F-55  system_prompt of every row in the no-persona probe file ==")
f = f"{R}/cap/run/probe/pc_position_holder_nopersona.jsonl"
for i, l in enumerate(open(f, encoding="utf-8")):
    r = json.loads(l)
    print(f"  row {i}: system_prompt={r.get('system_prompt')!r}  turn1_is_correct={r.get('turn1_is_correct')} "
          f"label={r.get('label')!r} kind={r.get('kind')!r}")
print("\nDONE W5B_labE")
