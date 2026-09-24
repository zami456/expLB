import json, statistics
d=json.load(open("C:/Users/user4/expLB/run-premise/premise_L40_mean_s42.json",encoding="utf-8"))
per=d["per_dataset"]
nl=[per[t]["n_lodo"] for t in per]
nm=[per[t]["n_lodo_match"] for t in per]
print("n_train block:", json.dumps(d.get("n_train")))
print("n_lodo       : mean %.0f  min %d  max %d"%(statistics.fmean(nl),min(nl),max(nl)))
print("n_lodo_match : mean %.0f  min %d  max %d"%(statistics.fmean(nm),min(nm),max(nm)))
for t in per:
    print("  %-34s n_lodo=%-6d n_lodo_match=%-5d"%(t,per[t]["n_lodo"],per[t]["n_lodo_match"]))
