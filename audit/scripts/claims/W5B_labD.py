"""W5B-ledger, lab pass D - READ-ONLY.  Two loose ends:
F-40: can the '111 chunks after 16.6 minutes / ~416 rows/min' reading be reconstructed from the
      8B log's progress bars (the chunk FILES were deleted before the 18:04 restart)?
F-50: when did the 3B arm actually finish (F-50 predicted ~21:20)?
"""
import re

R = "C:/Users/user4/expLB"


def readtext(p):
    b = open(p, "rb").read()
    return b.decode("utf-16", "replace") if b[:2] in (b"\xff\xfe", b"\xfe\xff") else b.decode("utf-8", "replace")


print("== F-40  8B log 17:14-17:18 progress bars (session counters) ==")
L = readtext(f"{R}/logs/scale_coords_8b.log").splitlines()
for i, l in enumerate(L, 1):
    if re.search(r"17:1[4-8]:", l) and re.search(r"\|\s*\d+/\d+ \[", l):
        m = re.search(r"(\w[\w-]*):\s+\d+%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<", l)
        if m:
            print(f"  {i:>6} {m.group(1)} {m.group(2)}/{m.group(3)} elapsed {m.group(4)}   ts={re.search(r'(\d\d:\d\d:\d\d)', l).group(1) if re.search(r'(\d\d:\d\d:\d\d)', l) else '?'}")
print("\n  last progress line of instructed-deception in the 17:00 run:")
for i, l in enumerate(L, 1):
    if "instructed-deception: 100%" in l and "85/85" in l:
        print(f"  {i:>6}: {l[:200]}")

print("\n== F-50  3B arm: dataset starts, completions and the final 'Done.' on 09-20/21 ==")
L3 = readtext(f"{R}/logs/scale_coords_3b.log").splitlines()
for i, l in enumerate(L3, 1):
    if re.search(r"2026-09-2[01].*(to do|already complete|Done\.|FAILED)", l):
        print(f"  {i:>6}: {l[:170]}")
print("\n  100% progress-bar lines (per dataset):")
for i, l in enumerate(L3, 1):
    if re.search(r": 100%\|", l) and re.search(r"\|\s*(\d+)/\1 \[", l.replace("|", "|")):
        pass
for i, l in enumerate(L3, 1):
    m = re.search(r"(\w[\w-]*): 100%\|[^|]*\|\s*(\d+)/(\d+) \[([\d:]+)<", l)
    if m and m.group(2) == m.group(3):
        print(f"  {i:>6} {m.group(1):28} {m.group(2)}/{m.group(3)} in {m.group(4)}")
print("\nDONE W5B_labD")
