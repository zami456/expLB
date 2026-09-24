"""What the audit actually covered. Reproduces every number in audit/COVERAGE_GAP.md.

The reports say "424 of 692 claims checked". That numerator mixes two different things:
inventory claims given a verdict, and claims raised during checking. Separating them is
the point of this script.
"""
import csv
import subprocess
from collections import Counter

INV = "audit/claims_inventory.csv"
CHK = "audit/claims.csv"


def main():
    inv = {r["id"]: r for r in csv.DictReader(open(INV, encoding="utf-8"))}
    chk = {r["id"] for r in csv.DictReader(open(CHK, encoding="utf-8"))}
    covered, raised, missed = chk & set(inv), chk - set(inv), set(inv) - chk

    print("=" * 78)
    print("1. COVERAGE — what the headline number hides")
    print("=" * 78)
    print(f"  inventory claims ({INV}) : {len(inv)}")
    print(f"  rows with a verdict ({CHK})  : {len(chk)}")
    print(f"  ...of those, IN the inventory     : {len(covered)}")
    print(f"  ...of those, raised while checking: {len(raised)}")
    print(f"  inventory claims with NO verdict   : {len(missed)}")
    print(f"\n  reported coverage : {len(chk)}/{len(inv)} = {100*len(chk)/len(inv):.0f}%")
    print(f"  REAL coverage     : {len(covered)}/{len(inv)} = {100*len(covered)/len(inv):.0f}%")

    print("\n" + "=" * 78)
    print("2. THE UNCHECKED CLAIMS, BY SOURCE")
    print("=" * 78)
    un = [inv[i] for i in missed]
    for k, v in Counter(r["source"] for r in un).most_common():
        flag = "   <-- the paper being submitted" if "draft" in k.lower() else ""
        print(f"  {v:5d}  {k}{flag}")
    print("\n  by status field:")
    for k, v in Counter(r["status"] or "(blank)" for r in un).most_common():
        print(f"  {v:5d}  {k}")

    print("\n" + "=" * 78)
    print("3. MAHDI'S BRANCH — does it carry results?")
    print("=" * 78)
    for desc, args in [
        ("commits beyond resumable-acts-cache",
         ["log", "--oneline", "origin/feat/resumable-acts-cache..origin/feat/llama-layer30"]),
        ("files those commits touch",
         ["diff", "--name-only", "origin/feat/resumable-acts-cache",
          "origin/feat/llama-layer30"]),
    ]:
        out = subprocess.run(["git"] + args, capture_output=True, text=True).stdout.strip()
        print(f"  {desc}:")
        for ln in out.splitlines():
            print(f"    {ln}")
    added = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=A", "asix",
         "origin/feat/llama-layer30"], capture_output=True, text=True).stdout
    n = sum(1 for ln in added.splitlines()
            if ln.rsplit(".", 1)[-1] in ("csv", "json", "npy", "npz", "jsonl"))
    print(f"  result files UNIQUE to that branch: {n}")
    print("  => the audit's Phase 0 read (branch_compare.md:308) was correct.")


if __name__ == "__main__":
    main()
