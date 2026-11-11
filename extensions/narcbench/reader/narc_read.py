"""NARCBench reader: Llama-3.3-70B on NDIF (nnsight 0.8.1), per row the reply-span mean and the end-of-turn vector.

Runs in `E:/aletheia/venv08` as `python -I narc_read.py ...` from `E:/narcbench/reader/`, a folder holding only
this file (nnsight serialises the caller's code; nothing of `deception_detection` is imported here). Inputs are the
rendered condition files of `conditions.py` (token ids, reply span [a, b), end-of-turn index b); outputs go to
`E:/narcbench/acts/llama33/<name>/<first row>_<n>.npz` with `rows`, `content` (n, len(CONTENT_LAYERS), H) and `eot`
(n, len(EOT_LAYERS), H), float32; a row with an empty reply has a NaN content mean.

Layer L is the output of decoder block L (hidden_states[L + 1]), read in ascending order; each pass keeps only the
last position's logits (logits_to_keep=1), because an early stop inside a multi-trace session fails on NDIF 0.8. Means are summed in float32 per layer on that layer's GPU and
moved to the CPU per layer (the 70B is sharded across GPUs). Batches are right-padded; the span mask is the only
source of the pooled tokens. Patterns follow `E:/qbank/code_src/src/assistant_axis_ndif/2_activations_ndif.py`,
the reader proven on nnsight 0.8 with Llama-3.3-70B.

Subcommands:
  selftest: a small random Llama (real Llama-3.3 vocabulary), `remote="local"`: session output == HF forward.
  smoke:    NDIF, label-blind: three rows read alone vs together (cos >= 0.9998), and the thesis's stored Alpaca rows
            re-read (cos >= 0.9998 at layers 20/36/40/44), from `alpaca_parity.npz` exported by `freeze.py`.
  read:     one condition file, optionally restricted to a run list; resumable; refuses unfrozen inputs.
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
ROOT = Path(r"E:\narcbench")
COND = ROOT / "rows" / "cond"
ACTS = ROOT / "acts" / "llama33"
LOGS = ROOT / "logs"
FREEZE = LOGS / "freeze.txt"
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
CONTENT_LAYERS = (20, 32, 33, 34, 35, 36, 37, 40, 44)
EOT_LAYERS = (32, 33, 34, 35, 36, 37, 40)
EOT_ID = 128009
PAD_ID = 128004  # <|finetune_right_pad_id|>
MIN_COS = 0.9998

log = logging.getLogger("narc_read")


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


def load_remote_llm():
    from nnsight import CONFIG, LanguageModel
    import nnsight.intervention.backends.remote as remote_backend

    remote_backend._READ_TIMEOUT = float(os.getenv("NARC_READ_TIMEOUT", "1800"))
    key = (os.getenv("NDIF_API_KEY") or "").strip()
    if not key:
        raise SystemExit("NDIF_API_KEY not set")
    host = (os.getenv("NDIF_HOST") or "https://api.ndif.us").strip()
    CONFIG.API.HOST = host if host.startswith("http") else "https://" + host
    CONFIG.API.APIKEY = key
    return LanguageModel(MODEL)


# ── one NDIF session: many traces, every row's means come back ───────────

def tensors(rows: list[dict]):
    import torch

    T = max(len(r["ids"]) for r in rows)
    B = len(rows)
    ids = torch.full((B, T), PAD_ID, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    cmask = torch.zeros((B, T), dtype=torch.float32)
    eot = torch.zeros(B, dtype=torch.long)
    for i, r in enumerate(rows):
        n, a, b = len(r["ids"]), r["a"], r["b"]
        e = r.get("eot", b)
        if not (0 <= a <= b <= n and 0 <= e < n):
            raise ValueError(f"row {r['row_id']}: span ({a},{b}) or eot {e} outside its {n} tokens")
        ids[i, :n] = torch.tensor(r["ids"], dtype=torch.long)
        attn[i, :n] = 1
        cmask[i, a:b] = 1.0
        eot[i] = e
    return ids, attn, cmask, eot


def session_read(llm, traces: list[list[dict]], hidden: int, remote, content_layers, eot_layers):
    """Read every row of every trace in one session. Returns (content, eot) float32 CPU tensors
    (rows in trace order, then row order), shapes (N, len(content_layers), H) and (N, len(eot_layers), H)."""
    import torch

    cl, el = list(content_layers), list(eot_layers)
    union = sorted(set(cl) | set(el))
    prepared, r0 = [], 0
    for rows in traces:
        ids, attn, cmask, eot = tensors(rows)
        prepared.append((ids, attn, cmask, eot, r0, r0 + len(rows)))
        r0 += len(rows)
    n_rows = r0
    # No early stop: on NDIF 0.8 a stop inside a multi-trace session breaks the sandbox (BrokenPipe /
    # "not enough values to unpack", measured 2026-10-20); logits_to_keep=1 keeps the 128k-vocabulary
    # head to the last position instead. (nnsight finds each `with` by its source line, so no comment
    # may sit between the loop header and the `with`.)
    with llm.session(remote=remote):
        content = torch.zeros(n_rows, len(cl), hidden, dtype=torch.float32).save()
        eot_out = torch.zeros(n_rows, len(el), hidden, dtype=torch.float32).save()
        for ids, attn, cmask, eot, s, e in prepared:
            with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=1):
                for L in union:  # ascending: an early block after a late one is an error
                    h = llm.model.layers[L].output
                    if L in cl:
                        m = cmask.to(h.device, torch.float32)
                        num = (h * m[:, :, None].to(h.dtype)).sum(dim=1, dtype=torch.float32)
                        den = m.sum(dim=1, dtype=torch.float32).clamp(min=1.0)[:, None]
                        content[s:e, cl.index(L)] += (num / den).cpu()
                    if L in el:
                        rows_idx = torch.arange(ids.shape[0], device=h.device)
                        eot_out[s:e, el.index(L)] += h[rows_idx, eot.to(h.device)].to(torch.float32).cpu()
    return getattr(content, "value", content), getattr(eot_out, "value", eot_out)


# ── planning, saving, resuming ───────────────────────────────────────────

def plan(rows: list[dict], budget: int, traces_per_session: int) -> list[list[list[dict]]]:
    """Longest first; a trace holds rows while (rows x longest) <= budget; a longer row goes alone."""
    rows = sorted(rows, key=lambda r: -len(r["ids"]))
    traces, cur = [], []
    for r in rows:
        if cur and (len(cur) + 1) * len(cur[0]["ids"]) > budget:
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


def save_chunk(folder: Path, rows: list[dict], content, eot) -> Path:
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["row_id"] for r in rows], dtype=np.int64)
    c = content.numpy().astype(np.float32)
    for i, r in enumerate(rows):
        if r["a"] == r["b"]:
            c[i] = np.nan
    path = folder / f"{int(ids.min()):06d}_{len(ids)}.npz"
    part = path.with_suffix(".partial")
    with open(part, "wb") as f:
        np.savez(f, rows=ids, content=c, eot=eot.numpy().astype(np.float32))
    os.replace(part, path)
    return path


def check_result(rows, content, eot) -> None:
    import torch

    filled = torch.tensor([r["a"] < r["b"] for r in rows])
    if not torch.isfinite(content[filled]).all() or not torch.isfinite(eot).all():
        raise RuntimeError("session returned non-finite values")
    if float(content[filled].abs().sum()) == 0 and filled.any():
        raise RuntimeError("session returned all-zero content means")
    if float(eot.abs().sum()) == 0:
        raise RuntimeError("session returned all-zero end-of-turn vectors")


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


def load_rows(name: str, runs_file: str | None, phases: str | None = None) -> list[dict]:
    path = COND / f"{name}.jsonl"
    keep = set(json.loads(Path(runs_file).read_text(encoding="utf-8"))) if runs_file else None
    want = set(phases.split(",")) if phases else None
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (keep is None or r["run_id"] in keep) and (want is None or r["phase"] in want):
                rows.append({k: r[k] for k in ("row_id", "run_id", "ids", "a", "b")})
    return rows


def cmd_read(args) -> int:
    setup_log(args.name)
    path = COND / f"{args.name}.jsonl"
    digest = sha256(path)
    if digest not in frozen_hashes():
        log.error("input %s (sha256 %s) is not in %s: refusing an unfrozen read", path.name, digest, FREEZE)
        return 2
    rows = load_rows(args.name, args.runs_file, args.phases)
    out = ACTS / args.name
    done = done_rows(out) if out.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    log.info("%s: %d rows selected, %d done, %d to read; budget %d tokens, %d traces/session",
             args.name, len(rows), len(rows) - len(todo), len(todo), args.budget, args.traces)
    if not todo:
        return 0
    llm = load_remote_llm()
    hidden = int(llm.config.hidden_size)
    budget, traces = args.budget, args.traces
    clean, waited, collisions, unknown = 0, 0.0, 0, 0
    unread_path = out / "unread.json"
    unread = json.loads(unread_path.read_text()) if unread_path.exists() else {}
    single_fail: dict[int, int] = {}
    todo = [r for r in todo if str(r["row_id"]) not in unread]
    while todo:
        sess = plan(todo, budget, traces)[0]
        flat = [r for tr in sess for r in tr]
        t0 = time.time()
        try:
            content, eot = with_watchdog(
                lambda: session_read(llm, sess, hidden, True, CONTENT_LAYERS, EOT_LAYERS), args.session_timeout)
            check_result(flat, content, eot)
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
                    log.error("three module-output collisions: another Llama reader is running; stopping")
                    return 5
                time.sleep(300)
                continue
            if kind in ("oom", "unknown") and len(flat) > 1:
                # one pass's activations are the usual cause (a 10k-token pass needs ~600 MB of MLP
                # intermediate): lower the pass budget while the largest pass is big, and only then the
                # number of passes per session (memory that builds up across passes)
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
        path = save_chunk(out, flat, content, eot)
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


def cmd_smoke(args) -> int:
    """Label-blind NDIF checks (G6): solo vs batched rows, and the stored Alpaca rows."""
    import numpy as np

    setup_log("smoke")
    llm = load_remote_llm()
    hidden = int(llm.config.hidden_size)
    rows = load_rows(args.name, None)[: args.n]
    solo_c, solo_e = with_watchdog(lambda: session_read(llm, [[r] for r in rows], hidden, True,
                                                        CONTENT_LAYERS, EOT_LAYERS), 3600)
    together_c, together_e = with_watchdog(lambda: session_read(llm, [rows], hidden, True,
                                                                CONTENT_LAYERS, EOT_LAYERS), 3600)
    batch = []
    for i, r in enumerate(rows):
        for j, L in enumerate(CONTENT_LAYERS):
            batch.append({"row": r["row_id"], "span": "content", "layer": L,
                          "cos": _cos(solo_c[i, j], together_c[i, j])})
        for j, L in enumerate(EOT_LAYERS):
            batch.append({"row": r["row_id"], "span": "eot", "layer": L, "cos": _cos(solo_e[i, j], together_e[i, j])})
    z = np.load(Path(__file__).with_name("alpaca_parity.npz"))
    parity_layers = [int(x) for x in z["layers"]]
    alp = [{"row_id": -1 - k, "ids": [int(t) for t in z[f"ids_{k}"]], "a": int(z["a"][k]), "b": int(z["b"][k]),
            "eot": len(z[f"ids_{k}"]) - 1} for k in range(len(z["a"]))]
    ac, _ = with_watchdog(lambda: session_read(llm, [alp], hidden, True, parity_layers, [parity_layers[0]]), 3600)
    stored = z["stored"]  # (rows, layers, H)
    parity = [{"alpaca_row": int(z["kept"][k]), "layer": L, "cos": _cos(ac[k, j], stored[k, j]),
               "rel_err": float(np.linalg.norm(ac[k, j].numpy() - stored[k, j]) / np.linalg.norm(stored[k, j]))}
              for k in range(len(alp)) for j, L in enumerate(parity_layers)]
    ok = min(b["cos"] for b in batch) >= MIN_COS and min(p["cos"] for p in parity) >= MIN_COS
    result = {"model": MODEL, "rows": [r["row_id"] for r in rows], "solo_vs_batched_min_cos": min(b["cos"] for b in batch),
              "alpaca_min_cos": min(p["cos"] for p in parity), "pass": ok, "batch": batch, "parity": parity}
    (LOGS / "g6_smoke.json").write_text(json.dumps(result, indent=1))
    log.info("smoke: solo vs batched min cos %.7f; Alpaca parity min cos %.7f; %s",
             result["solo_vs_batched_min_cos"], result["alpaca_min_cos"], "PASS" if ok else "FAIL")
    return 0 if ok else 1


def cmd_selftest(args) -> int:
    """Session output under remote='local' equals a plain HF forward, on a small random Llama."""
    import torch
    from nnsight import LanguageModel
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    setup_log("selftest")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    hf = LlamaForCausalLM(cfg).eval()
    llm = LanguageModel(hf, tokenizer=tok)
    gen = torch.Generator().manual_seed(1)
    rows = []
    for i, (n, a, b) in enumerate([(40, 10, 39), (25, 5, 24), (60, 30, 59), (12, 11, 11), (33, 2, 32)]):
        ids = torch.randint(1000, 120000, (n,), generator=gen).tolist()
        ids[0], ids[b] = 128000, EOT_ID
        rows.append({"row_id": i, "ids": ids, "a": a, "b": b})
    # hidden_states[-1] is the final norm's output, so the test reads blocks clear of the last one
    cl, el = (1, 3, 6), (2, 6)
    traces = [[rows[0], rows[1], rows[3]], [rows[2]], [rows[4], rows[1]]]
    content, eot = session_read(llm, traces, 64, "local", cl, el)
    flat = [r for tr in traces for r in tr]
    worst = 0.0
    with torch.no_grad():
        for i, r in enumerate(flat):
            hs = hf(torch.tensor([r["ids"]]), output_hidden_states=True).hidden_states
            for j, L in enumerate(cl):
                ref = hs[L + 1][0, r["a"]:r["b"]].mean(0) if r["b"] > r["a"] else torch.zeros(64)
                d = float((content[i, j] - ref).abs().max())
                log.info("  row %d content L%d diff %.3g", r["row_id"], L, d)
                worst = max(worst, d)
            for j, L in enumerate(el):
                d = float((eot[i, j] - hs[L + 1][0, r["b"]]).abs().max())
                log.info("  row %d eot L%d diff %.3g", r["row_id"], L, d)
                worst = max(worst, d)
    ok = worst < 1e-4
    log.info("selftest: max |session - HF forward| = %.3g over %d rows; %s", worst, len(flat), "PASS" if ok else "FAIL")
    (LOGS / "g6_selftest.json").write_text(json.dumps({"max_abs_diff": worst, "pass": ok}))
    return 0 if ok else 1


def main() -> int:
    setup_env()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("smoke")
    s.add_argument("--name", default="released__orig__core")
    s.add_argument("--n", type=int, default=3)
    r = sub.add_parser("read")
    r.add_argument("--name", required=True)
    r.add_argument("--runs_file", default=None)
    r.add_argument("--phases", default=None)
    r.add_argument("--budget", type=int, default=6000)
    r.add_argument("--min_budget", type=int, default=4096)
    r.add_argument("--traces", type=int, default=8)
    r.add_argument("--session_timeout", type=float, default=2700.0)
    r.add_argument("--max_wait", type=float, default=4 * 3600.0)
    args = p.parse_args()
    return {"selftest": cmd_selftest, "smoke": cmd_smoke, "read": cmd_read}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
