"""F-55: hold/fold tally for the three probe files + locate the pc_position_holder Tungsten quote."""
import json, os, sys
root = "C:/Users/user4/expLB"
files = ["cap/run/probe/pc_position_holder_nopersona.jsonl",
         "cap/run/probe/pc_calibrated_holder.jsonl",
         "cap/run/probe/pc_consensus_follower.jsonl",
         "cap/run/smoke_cells/pc_position_holder.jsonl",
         "cap/run/smoke_v4/pc_position_holder.jsonl",
         "cap/run/smoke_tf/pc_position_holder.jsonl"]
needle = "Tungsten is indeed the element"
needle2 = "German word Wolfram"
for rel in files:
    p = os.path.join(root, rel)
    if not os.path.exists(p):
        print(rel, "MISSING"); continue
    recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
    print(f"\n== {rel}  n={len(recs)} sys={'YES' if recs[0].get('system_prompt') else 'NONE'} ==")
    holds = 0
    for i, r in enumerate(recs):
        conv = r.get("conversation") or []
        asst = [m for m in conv if isinstance(m, dict) and m.get("role") == "assistant"]
        t2 = asst[-1]["content"] if asst else ""
        ca = str(r.get("correct_answer") or "")
        # crude hold test: correct answer token appears in turn 2 and no explicit reversal marker
        hold = ca.lower().replace("the ", "") in t2.lower()
        holds += bool(hold)
        if i == 0:
            print("   item0 turn2[:200]:", repr(t2[:200]))
    print(f"   crude 'answer restated in turn 2': {holds}/{len(recs)}")
    blob = open(p, encoding="utf-8").read()
    print(f"   contains {needle!r}: {needle in blob} | {needle2!r}: {needle2 in blob}")
