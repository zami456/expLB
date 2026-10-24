"""Label-blind NDIF diagnostic (2026-10-24): which basis sizes does the span reader's request get through with?"""
import sys
import time

sys.path.insert(0, r"E:\narcbench\verdict\reader")
import seq_read as SR  # noqa: E402
import verdict_read as VR  # noqa: E402

VR.setup_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402

rows, _, n_spans = SR.load_span_rows("qwen_attn_core")
row = min(rows, key=lambda r: len(r["ids"]))
llm = VR.load_remote_llm(VR.MODELS["qwen"]["id"])
layers = VR.MODELS["qwen"]["points"]
with np.load(SR.BASIS_FILE) as z:
    full = {L: z[f"B|L{L}"].astype(np.float32) for L in layers}
for dims in [int(x) for x in sys.argv[1:]]:
    SR.BASIS["B"] = {L: full[L][:dims] for L in layers}
    SR.BASIS["bt"] = torch.stack([torch.from_numpy(full[L][:dims].T.copy()) for L in layers]).to(torch.float16)
    mb = SR.BASIS["bt"].numel() * 2 / 1e6
    t0 = time.time()
    try:
        _, p = SR.session_project(llm, [[row]], 0, True, VR.MODELS["qwen"], 0, n_spans)
        print(f"dims {dims} ({mb:.2f} MB): OK in {time.time() - t0:.0f} s, shape {tuple(p.shape)}", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"dims {dims} ({mb:.2f} MB): FAILED in {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[-160:]}",
              flush=True)
