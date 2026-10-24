"""Label-blind NDIF diagnostic (2026-10-24): can a session build seeded random orthonormal bases on the server
(torch CPU generator + QR), and do they equal the same code run here?"""
import sys
import time

sys.path.insert(0, r"E:\narcbench\verdict\reader")
import seq_read as SR  # noqa: E402
import verdict_read as VR  # noqa: E402

VR.setup_env()
import numpy as np  # noqa: E402
import torch  # noqa: E402


def gen(seed: int, hidden: int = 5120, k: int = 3):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, k, generator=g, dtype=torch.float64))
    return q.to(torch.float32)


def session(llm, row, seeds):
    ids, attn, flat, S = SR.span_tensors([row], VR.MODELS["qwen"]["pad"], 1)
    with llm.session(remote=True):
        made = torch.zeros(len(seeds), 5120, 3).save()
        proj = torch.zeros(1, S, len(seeds), 3).save()
        for j, s in enumerate(seeds):
            g = torch.Generator().manual_seed(s)
            q, _ = torch.linalg.qr(torch.randn(5120, 3, generator=g, dtype=torch.float64))
            made[j] += q.to(torch.float32)
        with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=1, use_cache=False):
            h = llm.model.layers[28].output
            rows_idx = torch.arange(ids.shape[0], device=h.device)[:, None]
            x = h[rows_idx, flat.to(h.device)].to(torch.float32)
            for j in range(len(seeds)):
                proj[:, :, j] += (x @ made[j].to(h.device)).cpu()
    return getattr(made, "value", made), getattr(proj, "value", proj)


rows, _, _ = SR.load_span_rows("qwen_attn_core")
row = min(rows, key=lambda r: len(r["ids"]))
llm = VR.load_remote_llm(VR.MODELS["qwen"]["id"])
seeds = [60_000, 60_001]
t0 = time.time()
try:
    made, proj = session(llm, row, seeds)
    local = torch.stack([gen(s) for s in seeds])
    print(f"OK in {time.time() - t0:.0f} s; torch here {torch.__version__}; max |server - local| basis "
          f"{float((made - local).abs().max()):.3g}; proj shape {tuple(proj.shape)}; orthonormal "
          f"{float((made[0].T @ made[0] - torch.eye(3)).abs().max()):.2g}", flush=True)
except Exception as e:  # noqa: BLE001
    print(f"FAILED in {time.time() - t0:.0f} s: {type(e).__name__}: {str(e)[-300:]}", flush=True)
