"""Span reader for the attention probe (2026-10-24): Qwen3-32B on NDIF; every token of each row's spans, projected
on the server onto a frozen basis, comes back -- never the full hidden state.

Deployed next to `verdict_read.py` (unchanged, imported) in `E:/narcbench/verdict/reader/` and run there as
`python -I seq_read.py ...` in `E:/aletheia/venv08`. `read` is `verdict_read.cmd_read` itself (planning, OOM policy,
watchdog, resume, the freeze gate on the rows file) with three of its functions replaced in its module: the session
(`session_project`), the row loader (rows carry `spans`, a fixed number of [a, b) per file) and the masking. Rows
come from `qwen_attn_rows.py`; the basis is the first `SENT_ROWS` rows of `E:/narcbench/frozen/qwen_attn_basis.npz`
(`B|L<layer>`, (324, H)), sent as float16 and applied in float32: per layer, `h[span tokens] @ B^T`. Outputs: `acts/qwen/<name>/<first>_<n>.npz`
with `rows` and `points` = (n, spans, S, layers, D) float32, S the chunk's longest span, zero past each span's end.

Subcommands: selftest (tiny Qwen3, remote='local': session == HF forward @ B^T), smoke (GQ2), read, recheck.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_read as VR  # noqa: E402

BASIS_FILE = Path(r"E:\narcbench\frozen\qwen_attn_basis.npz")
# NDIF refuses a request body of ~1 MB or more (2026-10-24 diagnostics: 3 basis rows = 0.15 MB pass, 24 rows =
# 1.2 MB fail with HTTP 400 "error parsing the body"). So only the 3 persona rows (k = 3) are sent, and the null
# subspaces are built on the server, by `null_basis`'s code, from seeds: subspace i at layer L is the QR of a
# (hidden, 3) float64 Gaussian from torch.Generator().manual_seed(600000 + 1000 L + i) (identical to a local run,
# checked on NDIF). Per token the reader returns 3 + 3 N_NULL coordinates: persona first, then null 0, 1, ...
SENT_ROWS = 3
N_NULL = 100
SOURCES = {"qwen_attn_refl": ("qwen_reflection", 0, 0), "qwen_attn_verdict": ("qwen_verdict", 1, 2)}  # span, point
BASIS = {}
log = VR.log


def load_basis(path: Path = BASIS_FILE, layers=VR.MODELS["qwen"]["points"]) -> None:
    import numpy as np
    import torch

    with np.load(path) as z:
        BASIS["B"] = {L: z[f"B|L{L}"][:SENT_ROWS].astype(np.float32) for L in layers}
    BASIS["bt"] = torch.stack([torch.from_numpy(BASIS["B"][L].T.copy()) for L in layers]).to(torch.float16)


def null_basis(L: int, hidden: int, n_null: int = N_NULL):
    """(hidden, 3 n_null) float32: the null subspaces at layer L, as the session builds them on the server."""
    import torch

    cols = []
    for i in range(n_null):
        g = torch.Generator().manual_seed(600_000 + 1000 * L + i)
        q, _ = torch.linalg.qr(torch.randn(hidden, 3, generator=g, dtype=torch.float64))
        cols.append(q.to(torch.float32))
    return torch.cat(cols, dim=1)


def full_basis(L: int, j: int, hidden: int):
    """(hidden, 3 + 3 N_NULL): the sent persona rows (as the session applies them), then the server's nulls."""
    import torch

    return torch.cat([BASIS["bt"][j].to(torch.float32), null_basis(L, hidden)], dim=1)


# ── the session ──────────────────────────────────────────────────────────

def span_tensors(rows: list[dict], pad: int, n_spans: int):
    import torch

    T = max(len(r["ids"]) for r in rows)
    S = max([b - a for r in rows for a, b in r["spans"]] + [1])
    ids = torch.full((len(rows), T), pad, dtype=torch.long)
    attn = torch.zeros((len(rows), T), dtype=torch.long)
    pos = torch.zeros((len(rows), n_spans, S), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["ids"])
        ids[i, :n] = torch.tensor(r["ids"], dtype=torch.long)
        attn[i, :n] = 1
        for g, (a, b) in enumerate(r["spans"]):
            if not 0 <= a <= b <= n:
                raise ValueError(f"row {r['row_id']}: span ({a},{b}) outside its {n} tokens")
            pos[i, g, :b - a] = torch.arange(a, b)
    return ids, attn, pos.reshape(len(rows), n_spans * S), S


def session_project(llm, traces: list[list[dict]], hidden: int, remote, spec: dict, n_groups: int, n_spans: int):
    """`verdict_read.session_read`'s signature: returns (empty content, projections (N, spans, S, layers, D))."""
    import torch

    layers = list(spec["points"])
    bt = BASIS["bt"]
    n_null = N_NULL
    D = int(bt.shape[-1]) + 3 * n_null
    prepared, r0, smax = [], 0, 1
    for rows in traces:
        ids, attn, flat, S = span_tensors(rows, spec["pad"], n_spans)
        prepared.append((ids, attn, flat, S, r0, r0 + len(rows)))
        r0 += len(rows)
        smax = max(smax, S)
    # The null subspaces are built here, on the server, as `null_basis` builds them (the code is inlined: a session
    # runs its own source). (nnsight finds each `with` by its source line: no comment between a loop and its `with`.)
    with llm.session(remote=remote):
        proj = torch.zeros(r0, n_spans, smax, len(layers), D, dtype=torch.float32).save()
        basis = []
        for j, L in enumerate(layers):
            cols = [bt[j].to(torch.float32)]
            for i in range(n_null):
                g = torch.Generator().manual_seed(600_000 + 1000 * L + i)
                q, _ = torch.linalg.qr(torch.randn(hidden, 3, generator=g, dtype=torch.float64))
                cols.append(q.to(torch.float32))
            basis.append(torch.cat(cols, dim=1))
        for ids, attn, flat, S, s, e in prepared:
            with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=1, use_cache=False):
                for j, L in enumerate(layers):
                    h = llm.model.layers[L].output
                    rows_idx = torch.arange(ids.shape[0], device=h.device)[:, None]
                    y = h[rows_idx, flat.to(h.device)].to(torch.float32) @ basis[j].to(h.device)
                    proj[s:e, :, :S, j] += y.reshape(e - s, n_spans, S, D).cpu()
    return torch.zeros(r0, 0), getattr(proj, "value", proj)


def load_span_rows(name: str) -> tuple[list[dict], int, int]:
    """`verdict_read.load_rows`'s contract for span rows: (rows, 0 groups, number of spans)."""
    rows = []
    with open(VR.ROWS / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({"row_id": r["row_id"], "ids": r["ids"], "spans": r["spans"], "groups": []})
    shapes = {len(r["spans"]) for r in rows}
    if len(shapes) != 1:
        raise ValueError(f"{name}: rows differ in their number of spans: {shapes}")
    return rows, 0, shapes.pop()


def masked_spans(rows: list[dict], content, proj, n_groups: int, n_spans: int):
    """Zero past each span's end (padding gathered token 0)."""
    import numpy as np

    p = proj.detach().cpu().numpy().astype(np.float32)
    for i, r in enumerate(rows):
        for g, (a, b) in enumerate(r["spans"]):
            p[i, g, b - a:] = 0.0
    return np.zeros((len(rows), 0), np.float32), p


def install() -> None:
    """Point `verdict_read.cmd_read` at the span session; refuse a basis that is not frozen."""
    digest = VR.sha256(BASIS_FILE)
    if digest not in VR.frozen_hashes():
        raise SystemExit(f"{BASIS_FILE.name} (sha256 {digest}) is not in {VR.FREEZE}: refusing an unfrozen basis")
    load_basis()
    VR.session_read, VR.load_rows, VR.masked = session_project, load_span_rows, masked_spans


# ── checks ───────────────────────────────────────────────────────────────

def _cos(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _read(llm, rows, n_spans, solo=True):
    traces = [[r] for r in rows] if solo else [rows]
    hidden = int(llm.config.hidden_size)
    return masked_spans([r for t in traces for r in t],
                        *VR.with_watchdog(lambda: session_project(llm, traces, hidden, True, VR.MODELS["qwen"], 0,
                                                                  n_spans), 3600), 0, n_spans)[1]


def cmd_selftest(args) -> int:
    """remote='local' on a tiny Qwen3: every span token's projection == the HF forward's hidden state @ the full
    basis (the sent rows, then `null_basis`), with right padding, two spans per row of different lengths and two
    rows per trace."""
    import numpy as np
    import torch
    from nnsight import LanguageModel

    VR.setup_log("seq_selftest")
    tok, hf = VR._tiny("qwen")
    llm = LanguageModel(hf, tokenizer=tok)
    gen = torch.Generator().manual_seed(2)
    layers = (2, 6)
    B = {L: torch.linalg.qr(torch.randn(64, SENT_ROWS, generator=gen))[0].T.numpy() for L in layers}
    BASIS["B"], BASIS["bt"] = B, torch.stack([torch.from_numpy(B[L].T.copy()) for L in layers]).to(torch.float16)
    rows = []
    for i, (n, spans) in enumerate([(40, [[2, 9], [30, 40]]), (25, [[0, 25], [5, 6]]), (60, [[41, 59], [10, 11]])]):
        rows.append({"row_id": i, "ids": torch.randint(1000, 120000, (n,), generator=gen).tolist(), "spans": spans})
    spec = dict(VR.MODELS["qwen"], points=layers)
    traces = [[rows[0], rows[1]], [rows[2]]]
    _, p = masked_spans([r for t in traces for r in t], *session_project(llm, traces, 64, "local", spec, 0, 2), 0, 2)
    worst, cases = 0.0, 0
    with torch.no_grad():
        for i, r in enumerate([r for t in traces for r in t]):
            hs = hf(torch.tensor([r["ids"]]), output_hidden_states=True).hidden_states
            for g, (a, b) in enumerate(r["spans"]):
                for j, L in enumerate(layers):
                    ref = hs[L + 1][0, a:b].numpy() @ full_basis(L, j, 64).numpy()
                    worst = max(worst, float(np.abs(p[i, g, :b - a, j] - ref).max()))
                    assert not p[i, g, b - a:].any(), "padding is not zero"
                    cases += 1
    ok = worst < 1e-4
    log.info("seq selftest: max |session - HF @ B^T| = %.3g over %d span-layers; %s", worst, cases, "PASS" if ok else "FAIL")
    (VR.LOGS / "gq1_seq_selftest.json").write_text(json.dumps({"max_abs_diff": worst, "comparisons": cases, "pass": ok}))
    return 0 if ok else 1


def cmd_smoke(args) -> int:
    """GQ2 on NDIF: for `--n` rows of each Verdict file, the last token of the span that ends at a point of the
    single-token read equals that read @ B^T (cos >= MIN_COS, all 324 dims, each layer); Core rows read twice solo
    and once batched; the upload's session time recorded."""
    import numpy as np

    install()
    VR.setup_log("seq_smoke")
    llm = VR.load_remote_llm(VR.MODELS["qwen"]["id"])
    layers = VR.MODELS["qwen"]["points"]
    full = {L: full_basis(L, j, int(llm.config.hidden_size)).numpy().astype(np.float64) for j, L in enumerate(layers)}
    result, ok = {"basis_dims": int(full[layers[0]].shape[1])}, True
    for name, (src, g, q) in SOURCES.items():
        rows, _, n_spans = load_span_rows(name)
        meta = json.loads((VR.ROWS / f"{name}_meta.json").read_text(encoding="utf-8"))["rows"]
        few, src_ids = rows[: args.n], [m["row_id"] for m in meta[: args.n]]
        t0 = time.time()
        p = _read(llm, few, n_spans)
        secs = time.time() - t0
        saved = {}
        for f in sorted((VR.ACTS / "qwen" / src).glob("*.npz")):
            with np.load(f) as z:
                rr, pts = z["rows"], z["points"]
            saved.update({int(r): pts[i] for i, r in enumerate(rr) if int(r) in src_ids})
        cos = []
        for i, (r, sid) in enumerate(zip(few, src_ids)):
            a, b = r["spans"][g]
            for j, L in enumerate(layers):
                cos.append(_cos(p[i, g, b - a - 1, j], saved[sid][q, j].astype(np.float64) @ full[L]))
        result[name] = {"rows": [r["row_id"] for r in few], "min_cos_vs_point_read": min(cos), "seconds": round(secs, 1)}
        ok &= min(cos) >= VR.MIN_COS
    rows, _, n_spans = load_span_rows("qwen_attn_core")
    few = rows[: args.n]
    t0 = time.time()
    a1 = _read(llm, few, n_spans)
    secs = time.time() - t0
    a2, ab = _read(llm, few, n_spans), _read(llm, few, n_spans, solo=False)
    rep = [_cos(a1[i, 0, t, j], a2[i, 0, t, j]) for i, r in enumerate(few) for t in range(r["spans"][0][1] - r["spans"][0][0])
           for j in range(len(layers))]
    bat = [_cos(a1[i, 0, t, j], ab[i, 0, t, j]) for i, r in enumerate(few) for t in range(r["spans"][0][1] - r["spans"][0][0])
           for j in range(len(layers))]
    result["qwen_attn_core"] = {"rows": [r["row_id"] for r in few], "solo_repeat_min_cos": min(rep),
                                "solo_vs_batched_min_cos": min(bat), "seconds": round(secs, 1)}
    ok &= min(rep) >= VR.MIN_COS
    result["pass"] = bool(ok)
    (VR.LOGS / "gq2_seq_smoke.json").write_text(json.dumps(result, indent=1))
    log.info("seq smoke: %s", json.dumps(result))
    return 0 if ok else 1


def cmd_recheck(args) -> int:
    """After a read: its first `--n` rows read again solo and compared with the saved reads, files loaded once."""
    import numpy as np

    install()
    VR.setup_log(f"seq_recheck_{args.name}")
    rows, _, n_spans = load_span_rows(args.name)
    pick = rows[: args.n]
    want = {r["row_id"] for r in pick}
    saved = {}
    for f in sorted((VR.ACTS / "qwen" / args.name).glob("*.npz")):
        with np.load(f) as z:
            rr = z["rows"]
            if want & {int(x) for x in rr}:
                pts = z["points"]
                saved.update({int(r): pts[i] for i, r in enumerate(rr) if int(r) in want})
    llm = VR.load_remote_llm(VR.MODELS["qwen"]["id"])
    p = _read(llm, pick, n_spans)
    cos = [_cos(p[i, g, t, j], saved[r["row_id"]][g, t, j]) for i, r in enumerate(pick)
           for g, (a, b) in enumerate(r["spans"]) for t in range(b - a) for j in range(p.shape[3])]
    result = {"rows_file": args.name, "rows": sorted(want), "min_cos": min(cos), "pass": min(cos) >= VR.MIN_COS}
    (VR.LOGS / f"seq_recheck_{args.name}.json").write_text(json.dumps(result, indent=1))
    log.info("seq recheck %s: min cos %.7f; %s", args.name, min(cos), "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


def cmd_read(args) -> int:
    install()
    return VR.cmd_read(args)


def main() -> int:
    VR.setup_env()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("smoke")
    s.add_argument("--n", type=int, default=2)
    r = sub.add_parser("read")
    r.add_argument("--model", choices=("qwen",), default="qwen")
    r.add_argument("--name", required=True)
    r.add_argument("--budget", type=int, default=7000)
    r.add_argument("--min_budget", type=int, default=4096)
    r.add_argument("--traces", type=int, default=8)
    r.add_argument("--session_timeout", type=float, default=2700.0)
    r.add_argument("--max_wait", type=float, default=4 * 3600.0)
    r.add_argument("--batched", action="store_true", help="pack rows into traces (default: one row per trace)")
    c = sub.add_parser("recheck")
    c.add_argument("--name", required=True)
    c.add_argument("--n", type=int, default=2)
    args = p.parse_args()
    return {"selftest": cmd_selftest, "smoke": cmd_smoke, "read": cmd_read, "recheck": cmd_recheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
