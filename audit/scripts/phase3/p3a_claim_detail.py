import csv, sys
rows = {r["id"]: r for r in csv.DictReader(open("audit/claims.csv", encoding="utf-8"))}
for cid in sys.argv[1:]:
    r = rows.get(cid)
    if not r:
        print(f"### {cid}: NOT IN claims.csv\n"); continue
    print(f"### {cid}  [{r['verdict']}]  conf={r['confidence']}  imp={r['importance']}  batch={r['batch']}")
    print("CLAIM:", r["claim_plain"])
    print("STATED:", r["stated_numbers"])
    print("SRC:", r["source_ref"])
    print("EVID:", r["evidence_file"], "||", r["evidence_key"])
    print("L1:", r["level1_value"][:900])
    print("L2:", r["level2_value"][:900])
    print("ROBUST:", r["robustness"][:600])
    print("NOTES:", r["notes"][:1200])
    print()
