import json
from pathlib import Path
p = Path("C:/Users/user4/expLB/isarc_a.parquet")
print(p, p.stat().st_size, "B")
try:
    import pyarrow.parquet as pq
    t = pq.read_table(p)
    print("rows", t.num_rows, "cols", t.column_names)
    d = t.to_pydict()
    for c in t.column_names:
        vals = d[c][:3]
        print(f"  {c}: {vals}")
    if "dialect" in t.column_names:
        from collections import Counter
        print("  dialect counts:", Counter(d["dialect"]))
except Exception as e:
    print("pyarrow unavailable:", type(e).__name__, e)
    import pandas as pd
    df = pd.read_parquet(p)
    print(df.shape, list(df.columns))
    print(df.head(3).to_string()[:1200])
