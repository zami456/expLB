"""Verdict reader: Llama-3.3-70B or Qwen3-32B on NDIF (nnsight 0.8.1); per row, the mean over each span group and
the vector at each point. A fork of `narc_read.py` (same session, OOM policy, resume and watchdog).

Runs in `E:/aletheia/venv08` as `python -I verdict_read.py ...` from `E:/narcbench/verdict/reader/`, a folder
holding only this file and `alpaca_parity.npz` (nnsight serialises the caller's code; nothing of
`deception_detection` is imported here). Inputs are the row files of `verdict_rows.py`: token ids, `groups`
(a list of span groups, each a list of [a, b) spans whose tokens are pooled together) and `points` (token
indices; -1 for an absent point). Outputs go to `E:/narcbench/verdict/acts/<model>/<name>/<first row>_<n>.npz`
with `rows`, `content` (n, groups, len(CONTENT_LAYERS), H) and `points` (n, points, len(POINT_LAYERS), H),
float32; an empty group or an absent point is NaN.

Layer L is the output of decoder block L (hidden_states[L + 1]), read in ascending order; passes run with
use_cache=False and logits_to_keep=1 (no KV cache, the head on the last position only). Each row is read in a trace
of its own: batching moves single-token vectors slightly, solo reads reproduce exactly.

Subcommands:
  selftest: small random Llama and Qwen3 models (real vocabularies), `remote="local"`: session output == HF forward.
  smoke:    NDIF, label-blind: solo reads repeated (cos >= 0.9998), solo vs batched (recorded); the longest row
            alone (the budget); Llama: the thesis's stored Alpaca rows (cos >= 0.9998); Qwen: the AWQ check rows,
            saved for the scorer.
  read:     one row file; resumable; refuses unfrozen inputs.
  recheck:  after a read, its first rows read again against the saved reads (cos >= 0.9998; a redeploy check).
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path

ENV_FILE = Path(r"D:\expLB\liars-bench\.env")
ROOT = Path(r"E:\narcbench\verdict")
ROWS = ROOT / "rows"
ACTS = ROOT / "acts"
LOGS = ROOT / "logs"
FREEZE = Path(r"E:\narcbench\logs\freeze.txt")
MODELS = {
    "llama": {"id": "meta-llama/Llama-3.3-70B-Instruct", "content": (20, 32, 33, 34, 35, 36, 37, 40, 44),
              "points": (32, 33, 34, 35, 36, 37, 40), "pad": 128004},  # <|finetune_right_pad_id|>
    "qwen": {"id": "Qwen/Qwen3-32B", "content": (), "points": (26, 27, 28, 29, 30), "pad": 151643},  # <|endoftext|>
}
RELEASED_QWEN_TRANSFER = Path(r"E:\narcbench\data\activations\qwen3_32b\transfer\activations_gen.npz")
MIN_COS = 0.9998

log = logging.getLogger("verdict_read")


def setup_env() -> None:
    """Keys and transport fixes, before nnsight or transformers are imported."""
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    from dotenv import load_dotenv

    load_dotenv(ENV_FILE)
    os.environ.setdefault("NDIF_HOST", "https://api.ndif.us")
    os.environ.setdefault("HF_HOME", "E:/hf_cache")
    try:
        import certifi

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:  # noqa: BLE001
        pass
    for v in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        if os.getenv(v):
            os.environ["HF_TOKEN"] = os.environ["HF_HUB_TOKEN"] = os.environ[v].strip()
            break


def setup_log(name: str) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOGS / f"read_{name}.log", encoding="utf-8")):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.setLevel(logging.INFO)


def load_remote_llm(model_id: str):
    from nnsight import CONFIG, LanguageModel
    import nnsight.intervention.backends.remote as remote_backend

    remote_backend._READ_TIMEOUT = float(os.getenv("NARC_READ_TIMEOUT", "1800"))
    key = (os.getenv("NDIF_API_KEY") or "").strip()
    if not key:
        raise SystemExit("NDIF_API_KEY not set")
    host = (os.getenv("NDIF_HOST") or "https://api.ndif.us").strip()
    CONFIG.API.HOST = host if host.startswith("http") else "https://" + host
    CONFIG.API.APIKEY = key
    return LanguageModel(model_id)


# ── one NDIF session: many traces, every row's means and points come back ──

def tensors(rows: list[dict], pad: int, n_groups: int, n_points: int):
    import torch

    T = max(len(r["ids"]) for r in rows)
    B = len(rows)
    ids = torch.full((B, T), pad, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    gmask = torch.zeros((B, max(n_groups, 1), T), dtype=torch.float32)
    pidx = torch.zeros((B, max(n_points, 1)), dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r["ids"])
        if len(r["groups"]) != n_groups or len(r["points"]) != n_points:
            raise ValueError(f"row {r['row_id']}: {len(r['groups'])} groups / {len(r['points'])} points, file has "
                             f"{n_groups} / {n_points}")
        ids[i, :n] = torch.tensor(r["ids"], dtype=torch.long)
        attn[i, :n] = 1
        for g, spans in enumerate(r["groups"]):
            for a, b in spans:
                if not 0 <= a <= b <= n:
                    raise ValueError(f"row {r['row_id']}: span ({a},{b}) outside its {n} tokens")
                gmask[i, g, a:b] = 1.0
        for j, p in enumerate(r["points"]):
            if not -1 <= p < n:
                raise ValueError(f"row {r['row_id']}: point {p} outside its {n} tokens")
            pidx[i, j] = max(p, 0)
    return ids, attn, gmask, pidx


def session_read(llm, traces: list[list[dict]], hidden: int, remote, spec: dict, n_groups: int, n_points: int):
    """Read every row of every trace in one session. Returns (content, points) float32 CPU tensors (rows in trace
    order, then row order), shapes (N, groups, len(content layers), H) and (N, points, len(point layers), H)."""
    import torch

    cl, pl = list(spec["content"]), list(spec["points"])
    union = sorted(set(cl) | set(pl))
    prepared, r0 = [], 0
    for rows in traces:
        ids, attn, gmask, pidx = tensors(rows, spec["pad"], n_groups, n_points)
        prepared.append((ids, attn, gmask, pidx, r0, r0 + len(rows)))
        r0 += len(rows)
    n_rows = r0
    # No early stop (it breaks a multi-trace session on NDIF 0.8); logits_to_keep=1 and no KV cache instead.
    # (nnsight finds each `with` by its source line, so no comment may sit between the loop header and the `with`.)
    with llm.session(remote=remote):
        content = torch.zeros(n_rows, max(n_groups, 1), max(len(cl), 1), hidden, dtype=torch.float32).save()
        points = torch.zeros(n_rows, max(n_points, 1), max(len(pl), 1), hidden, dtype=torch.float32).save()
        for ids, attn, gmask, pidx, s, e in prepared:
            with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=1, use_cache=False):
                for L in union:  # ascending: an early block after a late one is an error
                    h = llm.model.layers[L].output
                    if L in cl:
                        for g in range(n_groups):
                            m = gmask[:, g].to(h.device, torch.float32)
                            num = (h * m[:, :, None].to(h.dtype)).sum(dim=1, dtype=torch.float32)
                            den = m.sum(dim=1, dtype=torch.float32).clamp(min=1.0)[:, None]
                            content[s:e, g, cl.index(L)] += (num / den).cpu()
                    if L in pl and n_points:
                        rows_idx = torch.arange(ids.shape[0], device=h.device)[:, None]
                        points[s:e, :, pl.index(L)] += h[rows_idx, pidx.to(h.device)].to(torch.float32).cpu()
    return getattr(content, "value", content), getattr(points, "value", points)


# ── planning, saving, resuming ───────────────────────────────────────────

def plan(rows: list[dict], budget: int, traces_per_session: int, solo: bool = True) -> list[list[list[dict]]]:
    """Longest first; a trace holds rows while (rows x longest) <= budget; a longer row goes alone. Solo (the
    default): one row per trace, because batching moves single-token vectors of ~2k-token rows by up to 4e-4 in
    cosine while solo reads reproduce exactly (GV2 diagnostic, 2026-10-22)."""
    rows = sorted(rows, key=lambda r: -len(r["ids"]))
    traces, cur = [], []
    for r in rows:
        if cur and (solo or (len(cur) + 1) * len(cur[0]["ids"]) > budget):
            traces.append(cur)
            cur = []
        cur.append(r)
    if cur:
        traces.append(cur)
    return [traces[i:i + traces_per_session] for i in range(0, len(traces), traces_per_session)]


def done_rows(folder: Path) -> set[int]:
    import numpy as np

    out = set()
    for f in folder.glob("*.npz"):
        with np.load(f) as z:
            out.update(int(x) for x in z["rows"])
    return out


def masked(rows: list[dict], content, points, n_groups: int, n_points: int):
    """NaN where a group is empty or a point absent; the placeholder axis dropped when a file has none."""
    import numpy as np

    c = content.detach().cpu().numpy().astype(np.float32)[:, :n_groups]
    p = points.detach().cpu().numpy().astype(np.float32)[:, :n_points]
    for i, r in enumerate(rows):
        for g, spans in enumerate(r["groups"]):
            if not any(b > a for a, b in spans):
                c[i, g] = np.nan
        for j, q in enumerate(r["points"]):
            if q < 0:
                p[i, j] = np.nan
    return c, p


def save_chunk(folder: Path, rows: list[dict], c, p) -> Path:
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["row_id"] for r in rows], dtype=np.int64)
    path = folder / f"{int(ids.min()):06d}_{len(ids)}.npz"
    part = path.with_suffix(".partial")
    with open(part, "wb") as f:
        np.savez(f, rows=ids, content=c, points=p)
    os.replace(part, path)
    return path


def check_result(c, p) -> None:
    import numpy as np

    for name, x in (("content", c), ("points", p)):
        if x.size == 0:
            continue
        valid = x[np.isfinite(x).all(axis=tuple(range(2, x.ndim)))]
        if np.isinf(x).any() or (valid.size and float(np.abs(valid).sum()) == 0):
            raise RuntimeError(f"session returned non-finite or all-zero {name}")
        if valid.size == 0 and x.shape[1]:
            raise RuntimeError(f"session returned no finite {name}")


def classify(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".lower()
    if "out of memory" in text or "outofmemory" in text:
        return "oom"
    if "outoforder" in text or "already ran past" in text:
        return "collision"
    if any(s in text for s in ("evicted", "deploying", "cant_accommodate", "connection", "timeout", "timed out",
                               "websocket", " 502", " 503", " 504", "temporarily", "not deployed", "queued")):
        return "transient"
    if isinstance(exc, (ValueError, TypeError, KeyError, IndexError, AttributeError, NameError)):
        return "fatal"
    return "unknown"


def tail(exc: BaseException) -> str:
    lines = [ln for ln in str(exc).strip().splitlines() if ln.strip()]
    return (lines[-1].strip() if lines else repr(exc))[:400]


def with_watchdog(fn, seconds: float):
    """Run fn in a thread; if it has not returned after `seconds`, exit the process (code 3) so the launcher
    relaunches it. A session stuck QUEUED on NDIF otherwise blocks forever."""
    box = {}

    def target():
        try:
            box["value"] = fn()
        except BaseException as e:  # noqa: BLE001
            box["error"] = e

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        log.error("session exceeded %.0f s; exiting for a relaunch", seconds)
        logging.shutdown()
        os._exit(3)
    if "error" in box:
        raise box["error"]
    return box["value"]


# ── commands ─────────────────────────────────────────────────────────────

def frozen_hashes() -> set[str]:
    if not FREEZE.exists():
        return set()
    text = FREEZE.read_text(encoding="utf-8-sig")
    return {w.strip().lower() for w in text.replace(",", " ").split() if len(w.strip()) == 64}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(name: str) -> tuple[list[dict], int, int]:
    rows = []
    with open(ROWS / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({k: r[k] for k in ("row_id", "ids", "groups", "points")})
    shapes = {(len(r["groups"]), len(r["points"])) for r in rows}
    if len(shapes) != 1:
        raise ValueError(f"{name}: rows differ in groups / points: {shapes}")
    n_groups, n_points = shapes.pop()
    return rows, n_groups, n_points


def cmd_read(args) -> int:
    spec = MODELS[args.model]
    setup_log(f"{args.model}_{args.name}")
    path = ROWS / f"{args.name}.jsonl"
    digest = sha256(path)
    if digest not in frozen_hashes():
        log.error("input %s (sha256 %s) is not in %s: refusing an unfrozen read", path.name, digest, FREEZE)
        return 2
    rows, n_groups, n_points = load_rows(args.name)
    if n_groups and not spec["content"]:
        log.error("%s has span groups but the %s reader reads no content layers", args.name, args.model)
        return 2
    out = ACTS / args.model / args.name
    done = done_rows(out) if out.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    log.info("%s on %s: %d rows, %d done, %d to read; budget %d tokens, %d traces/session", args.name, spec["id"],
             len(rows), len(rows) - len(todo), len(todo), args.budget, args.traces)
    if not todo:
        return 0
    llm = load_remote_llm(spec["id"])
    hidden = int(llm.config.hidden_size)
    budget, traces = args.budget, args.traces
    clean, waited, collisions, unknown = 0, 0.0, 0, 0
    unread_path = out / "unread.json"
    unread = json.loads(unread_path.read_text()) if unread_path.exists() else {}
    single_fail: dict[int, int] = {}
    todo = [r for r in todo if str(r["row_id"]) not in unread]
    while todo:
        sess = plan(todo, budget, traces, solo=not args.batched)[0]
        flat = [r for tr in sess for r in tr]
        t0 = time.time()
        try:
            content, points = with_watchdog(
                lambda: session_read(llm, sess, hidden, True, spec, n_groups, n_points), args.session_timeout)
            c, p = masked(flat, content, points, n_groups, n_points)
            check_result(c, p)
        except Exception as e:  # noqa: BLE001
            kind = classify(e)
            log.warning("session of %d rows (%d traces, longest %d) failed [%s]: %s: %s", len(flat), len(sess),
                        len(sess[0][0]["ids"]), kind, type(e).__name__, tail(e))
            if kind == "fatal":
                raise
            clean = 0
            if kind == "collision":
                collisions += 1
                if collisions >= 3:
                    log.error("three module-output collisions: another reader of this model is running; stopping")
                    return 5
                time.sleep(300)
                continue
            if kind in ("oom", "unknown") and len(flat) > 1:
                # lower the pass budget while the largest pass is big, and only then the passes per session
                largest = max(len(tr) * len(tr[0]["ids"]) for tr in sess)
                if largest > args.min_budget and budget > len(sess[0][0]["ids"]):
                    budget = max(len(sess[0][0]["ids"]), args.min_budget, largest // 2)
                elif traces > 1:
                    traces = max(1, traces // 2)
                else:
                    budget = max(len(sess[0][0]["ids"]), budget // 2)
                log.info("lowered to %d traces/session, budget %d tokens", traces, budget)
                if kind == "unknown":
                    unknown += 1
                    if unknown >= 6:
                        log.error("six unexplained failures in a row; stopping for inspection")
                        return 6
                continue
            if kind == "oom" and len(flat) == 1:
                rid = flat[0]["row_id"]
                single_fail[rid] = single_fail.get(rid, 0) + 1
                if single_fail[rid] >= 4:
                    unread[str(rid)] = f"out of memory alone ({len(flat[0]['ids'])} tokens)"
                    unread_path.parent.mkdir(parents=True, exist_ok=True)
                    unread_path.write_text(json.dumps(unread, indent=1))
                    todo = [r for r in todo if r["row_id"] != rid]
                    log.error("row %d left unread after 4 solo out-of-memory failures", rid)
                    continue
                time.sleep(60 * single_fail[rid])
                continue
            delay = min(60.0 * 2 ** min(int(waited // 600), 3), 600.0)
            waited += delay
            if waited > args.max_wait:
                log.error("waited %.0f s on transient failures; stopping for a relaunch", waited)
                return 4
            time.sleep(delay)
            continue
        path = save_chunk(out, flat, c, p)
        got = {r["row_id"] for r in flat}
        todo = [r for r in todo if r["row_id"] not in got]
        clean += 1
        waited, collisions, unknown = 0.0, 0, 0
        log.info("saved %s (%d rows, %.0f s); %d rows left", path.name, len(flat), time.time() - t0, len(todo))
        if clean >= 3 and traces < args.traces:  # the pass budget is not raised back: it is the replica's memory
            traces = min(args.traces, traces * 2)
            clean = 0
            log.info("raised back to %d traces/session (budget stays %d tokens)", traces, budget)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    log.info("%s complete", args.name)
    return 0


def _cos(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _finite_pairs(x, y):
    import numpy as np

    for idx in np.ndindex(*x.shape[:3]):
        if np.isfinite(x[idx]).all() and np.isfinite(y[idx]).all():
            yield idx, _cos(x[idx], y[idx])


def cmd_smoke(args) -> int:
    """Label-blind NDIF checks (GV2) for one model: solo reads repeated (reads run solo), solo vs batched (recorded),
    the longest row alone, and Alpaca parity (Llama) or the AWQ check rows (Qwen)."""
    import numpy as np

    spec = MODELS[args.model]
    setup_log(f"smoke_{args.model}")
    llm = load_remote_llm(spec["id"])
    hidden = int(llm.config.hidden_size)
    rows, ng, npts = load_rows(args.name)
    few = sorted(rows, key=lambda r: len(r["ids"]))[: args.n]
    reads = [masked(few, *with_watchdog(lambda t=t: session_read(llm, t, hidden, True, spec, ng, npts), 3600), ng, npts)
             for t in ([[r] for r in few], [[r] for r in few], [few])]
    repeat = [c for x, y in zip(reads[0], reads[1]) for _, c in _finite_pairs(x, y)]
    batch = [c for x, y in zip(reads[0], reads[2]) for _, c in _finite_pairs(x, y)]
    longest = max(rows, key=lambda r: len(r["ids"]))
    t0 = time.time()
    lc, lp = masked([longest], *with_watchdog(lambda: session_read(llm, [[longest]], hidden, True, spec, ng, npts),
                                               3600), ng, npts)
    result = {"model": spec["id"], "rows_file": args.name, "rows": [r["row_id"] for r in few],
              "solo_repeat_min_cos": min(repeat), "solo_vs_batched_min_cos": min(batch),
              "longest_row": {"row_id": longest["row_id"], "tokens": len(longest["ids"]),
                              "seconds": round(time.time() - t0, 1),
                              "finite": bool(np.isfinite(lp).all() and (lc.size == 0 or np.isfinite(lc[np.isfinite(lc)]).all()))}}
    ok = result["solo_repeat_min_cos"] >= MIN_COS and result["longest_row"]["finite"]
    if args.model == "llama":
        z = np.load(Path(__file__).with_name("alpaca_parity.npz"))
        parity_layers = [int(x) for x in z["layers"]]
        alp = [{"row_id": -1 - k, "ids": [int(t) for t in z[f"ids_{k}"]], "groups": [[[int(z["a"][k]), int(z["b"][k])]]],
                "points": [len(z[f"ids_{k}"]) - 1]} for k in range(len(z["a"]))]
        pspec = dict(spec, content=tuple(parity_layers), points=(parity_layers[0],))
        ac, _ = with_watchdog(lambda: session_read(llm, [alp], hidden, True, pspec, 1, 1), 3600)
        stored = z["stored"]
        parity = [{"alpaca_row": int(z["kept"][k]), "layer": L, "cos": _cos(ac[k, 0, j], stored[k, j])}
                  for k in range(len(alp)) for j, L in enumerate(parity_layers)]
        result["alpaca_min_cos"] = min(p["cos"] for p in parity)
        ok &= result["alpaca_min_cos"] >= MIN_COS
    else:
        awq, _, _ = load_rows("qwen_awq_check")
        meta = [json.loads(line)["meta_index"] for line in open(ROWS / "qwen_awq_check.jsonl", encoding="utf-8")]
        _, qp = masked(awq, *with_watchdog(lambda: session_read(llm, [[r] for r in awq], hidden, True, spec, 0, 1),
                                           3600), 0, 1)
        out = ACTS / "qwen" / "qwen_awq_check"
        save_chunk(out, awq, np.zeros((len(awq), 0), np.float32), qp)
        with np.load(RELEASED_QWEN_TRANSFER) as z:
            released = np.stack([z[f"layer_{L}"][meta] for L in spec["points"]], axis=1)  # (rows, layers, H)
        cos = [_cos(qp[i, 0, j], released[i, j]) for i in range(len(awq)) for j in range(len(spec["points"]))]
        result["awq_vs_bf16_cos"] = {"min": min(cos), "median": float(np.median(cos)), "mean": float(np.mean(cos))}
    result["pass"] = bool(ok)
    (LOGS / f"gv2_smoke_{args.model}.json").write_text(json.dumps(result, indent=1))
    log.info("smoke %s: %s", args.model, json.dumps({k: v for k, v in result.items() if k != "rows"}))
    return 0 if ok else 1


def cmd_recheck(args) -> int:
    """After a file is read: its first rows read again and compared with the saved reads (a redeploy check)."""
    import numpy as np

    spec = MODELS[args.model]
    setup_log(f"recheck_{args.model}_{args.name}")
    rows, ng, npts = load_rows(args.name)
    saved = {}
    for f in sorted((ACTS / args.model / args.name).glob("*.npz")):
        with np.load(f) as z:
            saved.update({int(r): (z["content"][i], z["points"][i]) for i, r in enumerate(z["rows"])})
    pick = [r for r in rows if r["row_id"] in saved][: args.n]
    llm = load_remote_llm(spec["id"])
    c, p = masked(pick, *with_watchdog(lambda: session_read(llm, [[r] for r in pick], int(llm.config.hidden_size), True,
                                                             spec, ng, npts), 3600), ng, npts)
    old_c = np.stack([saved[r["row_id"]][0] for r in pick])
    old_p = np.stack([saved[r["row_id"]][1] for r in pick])
    cos = [x for a, b in ((c, old_c), (p, old_p)) if a.size for _, x in _finite_pairs(a, b)]
    result = {"model": spec["id"], "rows_file": args.name, "rows": [r["row_id"] for r in pick], "min_cos": min(cos),
              "pass": min(cos) >= MIN_COS}
    (LOGS / f"recheck_{args.model}_{args.name}.json").write_text(json.dumps(result, indent=1))
    log.info("recheck %s %s: min cos %.7f; %s", args.model, args.name, min(cos), "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


def _tiny(model: str):
    import torch
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODELS[model]["id"], local_files_only=True)
    if model == "llama":
        from transformers import LlamaConfig, LlamaForCausalLM

        cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
        return tok, LlamaForCausalLM(cfg).eval()
    from transformers import Qwen3Config, Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=1024)
    return tok, Qwen3ForCausalLM(cfg).eval()


def cmd_selftest(args) -> int:
    """Session output under remote='local' equals a plain HF forward, for both architectures, with several span
    groups (one empty), several points (one absent) and right padding."""
    import numpy as np
    import torch
    from nnsight import LanguageModel

    setup_log("selftest")
    worst, cases = 0.0, 0
    for model in ("llama", "qwen"):
        tok, hf = _tiny(model)
        llm = LanguageModel(hf, tokenizer=tok)
        gen = torch.Generator().manual_seed(1)
        rows = []
        for i, (n, groups, points) in enumerate([
                (40, [[[2, 5], [10, 15]], [[20, 39]], []], [4, 14, -1]),
                (25, [[[5, 24]], [], [[1, 2]]], [24, -1, 0]),
                (60, [[[30, 40], [41, 59]], [[0, 1]], [[50, 59]]], [59, 30, 12]),
                (12, [[], [[3, 4]], [[11, 12]]], [-1, -1, 11])]):
            ids = torch.randint(1000, 120000, (n,), generator=gen).tolist()
            rows.append({"row_id": i, "ids": ids, "groups": groups, "points": points})
        # hidden_states[-1] is the final norm's output, so the test reads blocks clear of the last one
        spec = dict(MODELS[model], content=(1, 3, 6), points=(2, 6))
        traces = [[rows[0], rows[1]], [rows[2]], [rows[3], rows[0]]]
        content, points = session_read(llm, traces, 64, "local", spec, 3, 3)
        flat = [r for tr in traces for r in tr]
        c, p = masked(flat, content, points, 3, 3)
        with torch.no_grad():
            for i, r in enumerate(flat):
                hs = hf(torch.tensor([r["ids"]]), output_hidden_states=True).hidden_states
                for g, spans in enumerate(r["groups"]):
                    idx = [t for a, b in spans for t in range(a, b)]
                    for j, L in enumerate(spec["content"]):
                        if not idx:
                            assert np.isnan(c[i, g, j]).all(), "an empty group is not NaN"
                            continue
                        ref = hs[L + 1][0, idx].mean(0).numpy()
                        worst = max(worst, float(np.abs(c[i, g, j] - ref).max()))
                        cases += 1
                for q, pos in enumerate(r["points"]):
                    for j, L in enumerate(spec["points"]):
                        if pos < 0:
                            assert np.isnan(p[i, q, j]).all(), "an absent point is not NaN"
                            continue
                        worst = max(worst, float(np.abs(p[i, q, j] - hs[L + 1][0, pos].numpy()).max()))
                        cases += 1
        log.info("selftest %s done", model)
    ok = worst < 1e-4
    log.info("selftest: max |session - HF forward| = %.3g over %d comparisons; %s", worst, cases, "PASS" if ok else "FAIL")
    (LOGS / "gv2_selftest.json").write_text(json.dumps({"max_abs_diff": worst, "comparisons": cases, "pass": ok}))
    return 0 if ok else 1


def main() -> int:
    setup_env()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("smoke")
    s.add_argument("--model", choices=tuple(MODELS), required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--n", type=int, default=3)
    r = sub.add_parser("read")
    r.add_argument("--model", choices=tuple(MODELS), required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--budget", type=int, default=7000)
    r.add_argument("--min_budget", type=int, default=4096)
    r.add_argument("--traces", type=int, default=8)
    r.add_argument("--session_timeout", type=float, default=2700.0)
    r.add_argument("--max_wait", type=float, default=4 * 3600.0)
    r.add_argument("--batched", action="store_true", help="pack rows into traces (default: one row per trace)")
    c = sub.add_parser("recheck")
    c.add_argument("--model", choices=tuple(MODELS), required=True)
    c.add_argument("--name", required=True)
    c.add_argument("--n", type=int, default=2)
    args = p.parse_args()
    return {"selftest": cmd_selftest, "smoke": cmd_smoke, "read": cmd_read, "recheck": cmd_recheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
