"""Emotion-coordinate reader: Llama-3.3-70B on NDIF (nnsight 0.8.1). A fork of the Verdict study's
`scripts/narcbench/reader/verdict_read.py` (same session layout, OOM policy, resume and watchdog), with two
additions: teacher-forced log-probs (`logprob`) and one-off sampling (`generate`).

Runs in `E:/aletheia/venv08` as `python -I ec_read.py ...`. Nothing of `deception_detection` is imported: nnsight
serialises the caller's code. Row files are written by `emotion_coords/rows.py` to `data/emotion_coords/rows/`:

  read     {"row_id", "ids", "groups", "points"}: per row, the mean over each span group ([a, b) spans pooled
           together) and the vector at each point (-1 = absent), at every layer in LAYERS. Output
           `acts/<name>/<first row>_<n>.npz` with `rows`, `content` (n, groups, layers, H), `points`
           (n, points, layers, H), float32; an empty group or absent point is NaN.
  logprob  {"row_id", "ids", "target_from"}: log p(ids[t] | ids[:t]) for t = target_from .. len(ids) - 1, under
           teacher forcing, gathered on the server (logits only for the scored positions). Output
           `acts/<name>/...npz` with `rows`, `logprobs` (n, longest) NaN-padded.
  generate {"row_id", "ids", "max_new_tokens", "temperature", "top_p"}: sampled continuations, written to
           `acts/<name>/generations.jsonl`.
  recheck  re-reads a finished `read` file's first rows and compares (cos >= 0.9998: a redeploy check).
  selftest a small random Llama under remote="local": session output == HF forward, for reads and log-probs.

Layer L is the output of decoder block L (hidden_states[L + 1]); passes run with use_cache=False and
logits_to_keep=1 (reads) or the number of scored positions (logprob). Rows are read one per trace by default
(batching moves single-token vectors slightly); `--batched` packs rows into traces up to the token budget.
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

HERE = Path(__file__).resolve()
DD = HERE.parents[2]
DATA = DD / "data" / "emotion_coords"
ROWS, ACTS, LOGS = DATA / "rows", DATA / "acts", DATA / "logs"
ENV_FILE = HERE.parents[5] / ".env"  # liars-bench/.env
MODEL = {"id": "meta-llama/Llama-3.3-70B-Instruct", "layers": (20, 28, 40, 52), "pad": 128004}  # <|finetune_right_pad_id|>
MIN_COS = 0.9998

log = logging.getLogger("ec_read")


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
    if Path("E:/hf_cache").exists():  # where the earlier NDIF reads cached Llama's config and tokenizer
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

    remote_backend._READ_TIMEOUT = float(os.getenv("EC_READ_TIMEOUT", "1800"))
    key = (os.getenv("NDIF_API_KEY") or "").strip()
    if not key:
        raise SystemExit("NDIF_API_KEY not set")
    host = (os.getenv("NDIF_HOST") or "https://api.ndif.us").strip()
    CONFIG.API.HOST = host if host.startswith("http") else "https://" + host
    CONFIG.API.APIKEY = key
    return LanguageModel(MODEL["id"])


# ── one NDIF session: many traces ──

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


def session_read(llm, traces: list[list[dict]], hidden: int, remote, layers, n_groups: int, n_points: int):
    """Every row of every trace in one session. Returns (content, points) float32 CPU tensors, rows in trace order
    then row order, shapes (N, groups, layers, H) and (N, points, layers, H)."""
    import torch

    layers = list(layers)
    prepared, r0 = [], 0
    for rows in traces:
        ids, attn, gmask, pidx = tensors(rows, MODEL["pad"], n_groups, n_points)
        prepared.append((ids, attn, gmask, pidx, r0, r0 + len(rows)))
        r0 += len(rows)
    n_rows = r0
    # No early stop (it breaks a multi-trace session on NDIF 0.8); logits_to_keep=1 and no KV cache instead.
    # (nnsight finds each `with` by its source line, so no comment may sit between the loop header and the `with`.)
    with llm.session(remote=remote):
        content = torch.zeros(n_rows, max(n_groups, 1), len(layers), hidden, dtype=torch.float32).save()
        points = torch.zeros(n_rows, max(n_points, 1), len(layers), hidden, dtype=torch.float32).save()
        for ids, attn, gmask, pidx, s, e in prepared:
            with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=1, use_cache=False):
                for j, L in enumerate(layers):  # ascending: an early block after a late one is an error
                    h = llm.model.layers[L].output
                    for g in range(n_groups):
                        m = gmask[:, g].to(h.device, torch.float32)
                        num = (h * m[:, :, None].to(h.dtype)).sum(dim=1, dtype=torch.float32)
                        den = m.sum(dim=1, dtype=torch.float32).clamp(min=1.0)[:, None]
                        content[s:e, g, j] += (num / den).cpu()
                    if n_points:
                        rows_idx = torch.arange(ids.shape[0], device=h.device)[:, None]
                        points[s:e, :, j] += h[rows_idx, pidx.to(h.device)].to(torch.float32).cpu()
    return getattr(content, "value", content), getattr(points, "value", points)


def session_logprob(llm, rows: list[dict], remote):
    """Teacher-forced log-probs of ids[target_from:] for each row, one row per trace, all in one session.
    Returns a float32 CPU tensor (N, longest), NaN beyond each row's targets."""
    import torch

    width = max(len(r["ids"]) - r["target_from"] for r in rows)
    prepared = []
    for i, r in enumerate(rows):
        ids = torch.tensor([r["ids"]], dtype=torch.long)
        n_targets = len(r["ids"]) - r["target_from"]
        targets = torch.tensor(r["ids"][r["target_from"]:], dtype=torch.long)
        prepared.append((i, ids, torch.ones_like(ids), n_targets + 1, n_targets, targets))
    # (nnsight finds each `with` by its source line: the loop header must lead straight into it.)
    with llm.session(remote=remote):
        out = torch.full((len(rows), width), float("nan"), dtype=torch.float32).save()
        for i, ids, attn, keep, n_targets, targets in prepared:
            with llm.trace({"input_ids": ids, "attention_mask": attn}, logits_to_keep=keep, use_cache=False):
                logits = llm.lm_head.output
                lp = logits[0, :-1].to(torch.float32).log_softmax(dim=-1)
                picked = lp.gather(-1, targets.to(lp.device)[:, None])[:, 0]
                out[i, :n_targets] = picked.cpu()
    return getattr(out, "value", out).detach()


# ── planning, saving, resuming ──

def plan(rows: list[dict], budget: int, traces_per_session: int, solo: bool = True) -> list[list[list[dict]]]:
    """Longest first; a trace holds rows while (rows x longest) <= budget; a longer row goes alone. Solo (the
    default): one row per trace."""
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


def save_chunk(folder: Path, rows: list[dict], **arrays) -> Path:
    import numpy as np

    folder.mkdir(parents=True, exist_ok=True)
    ids = np.array([r["row_id"] for r in rows], dtype=np.int64)
    path = folder / f"{int(ids.min()):09d}_{len(ids)}.npz"
    part = path.with_suffix(".partial")
    with open(part, "wb") as f:
        np.savez(f, rows=ids, **arrays)
    os.replace(part, path)
    return path


def check_result(**arrays) -> None:
    import numpy as np

    for name, x in arrays.items():
        if x.size == 0:
            continue
        axes = tuple(range(2, x.ndim)) if x.ndim > 2 else (1,)
        valid = x[np.isfinite(x).all(axis=axes)] if x.ndim > 2 else x[np.isfinite(x)]
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(name: str, keys=("row_id", "ids", "groups", "points")) -> list[dict]:
    with open(ROWS / f"{name}.jsonl", encoding="utf-8") as f:
        return [{k: r[k] for k in keys if k in r} for r in map(json.loads, f)]


def shape_of(rows: list[dict]) -> tuple[int, int]:
    shapes = {(len(r["groups"]), len(r["points"])) for r in rows}
    if len(shapes) != 1:
        raise ValueError(f"rows differ in groups / points: {shapes}")
    return shapes.pop()


# ── the resumable loop shared by `read` and `logprob` ──

def run_loop(args, rows: list[dict], work, out: Path, solo: bool, done_name: str = "DONE") -> int:
    """Read `rows` session by session with `work(session_traces) -> dict of arrays`, saving each session's chunk.
    Same failure policy as the Verdict reader: halve on OOM, back off on transient failures, stop on repeated
    unexplained ones."""
    budget, traces = args.budget, args.traces
    clean, waited, collisions, unknown = 0, 0.0, 0, 0
    unread_path = out / "unread.json"
    unread = json.loads(unread_path.read_text()) if unread_path.exists() else {}
    single_fail: dict[int, int] = {}
    todo = [r for r in rows if str(r["row_id"]) not in unread]
    n_requests, t_start = 0, time.time()
    while todo:
        sess = plan(todo, budget, traces, solo=solo)[0]
        flat = [r for tr in sess for r in tr]
        t0 = time.time()
        n_requests += 1
        try:
            arrays = with_watchdog(lambda: work(sess, flat), args.session_timeout)
            check_result(**arrays)
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
        path = save_chunk(out, flat, **arrays)
        got = {r["row_id"] for r in flat}
        todo = [r for r in todo if r["row_id"] not in got]
        clean += 1
        waited, collisions, unknown = 0.0, 0, 0
        log.info("saved %s (%d rows, %d tokens, %.0f s); %d rows left", path.name, len(flat),
                 sum(len(r["ids"]) for r in flat), time.time() - t0, len(todo))
        if clean >= 3 and traces < args.traces:  # the pass budget is not raised back: it is the replica's memory
            traces = min(args.traces, traces * 2)
            clean = 0
            log.info("raised back to %d traces/session (budget stays %d tokens)", traces, budget)
    stats = {"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "requests_this_run": n_requests,
             "seconds_this_run": round(time.time() - t_start, 1)}
    (out / done_name).write_text(json.dumps(stats))
    log.info("%s complete: %s", out.name, stats)
    return 0


# ── commands ──

def cmd_read(args) -> int:
    """`--shard K/N` reads only the rows at positions K mod N of the rows file, so N processes can split a file
    without overlap (NDIF's result downloads are slow per connection, not per client)."""
    k, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= k < n:
        raise SystemExit(f"bad --shard {args.shard}")
    setup_log(args.name if n == 1 else f"{args.name}_shard{k}of{n}")
    rows = load_rows(args.name)
    n_groups, n_points = shape_of(rows)
    out = ACTS / args.name
    done = done_rows(out) if out.exists() else set()
    todo = [r for i, r in enumerate(rows) if i % n == k and r["row_id"] not in done]
    log.info("%s (sha256 %s): %d rows, %d done, %d to read in shard %d/%d; layers %s; budget %d tokens, "
             "%d traces/session, %s", args.name, sha256(ROWS / f"{args.name}.jsonl")[:12], len(rows), len(done),
             len(todo), k, n, MODEL["layers"], args.budget, args.traces, "batched" if args.batched else "solo")
    if not todo:
        return 0
    llm = load_remote_llm()
    hidden = int(llm.config.hidden_size)

    def work(sess, flat):
        content, points = session_read(llm, sess, hidden, True, MODEL["layers"], n_groups, n_points)
        c, p = masked(flat, content, points, n_groups, n_points)
        return {"content": c, "points": p}

    return run_loop(args, todo, work, out, solo=not args.batched,
                    done_name="DONE" if n == 1 else f"DONE.shard{k}of{n}")


def cmd_logprob(args) -> int:
    import numpy as np

    setup_log(args.name)
    rows = load_rows(args.name, ("row_id", "ids", "target_from"))
    out = ACTS / args.name
    done = done_rows(out) if out.exists() else set()
    todo = [r for r in rows if r["row_id"] not in done]
    log.info("%s logprob: %d rows, %d to read", args.name, len(rows), len(todo))
    if not todo:
        return 0
    llm = load_remote_llm()

    def work(sess, flat):
        lp = session_logprob(llm, flat, True).numpy().astype(np.float32)
        return {"logprobs": lp}

    return run_loop(args, todo, work, out, solo=True)


def cmd_generate(args) -> int:
    import torch

    setup_log(args.name)
    rows = load_rows(args.name, ("row_id", "ids", "max_new_tokens", "temperature", "top_p"))
    out = ACTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / "generations.jsonl"
    done = {json.loads(line)["row_id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
    llm = load_remote_llm()
    tok = llm.tokenizer
    for r in rows:
        if r["row_id"] in done:
            continue
        ids = torch.tensor([r["ids"]], dtype=torch.long)
        t0 = time.time()
        for attempt in range(args.retries):
            try:
                with llm.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=r["max_new_tokens"],
                                  do_sample=r["temperature"] > 0, temperature=r["temperature"], top_p=r["top_p"],
                                  pad_token_id=tok.pad_token_id or MODEL["pad"], remote=True):
                    gen = llm.generator.output.save()
                gen = getattr(gen, "value", gen)
                break
            except Exception as e:  # noqa: BLE001
                log.warning("generate row %s attempt %d failed [%s]: %s", r["row_id"], attempt + 1, classify(e), tail(e))
                if attempt + 1 == args.retries:
                    raise
                time.sleep(60 * (attempt + 1))
        new = gen[0, len(r["ids"]):].tolist()
        record = {"row_id": r["row_id"], "seconds": round(time.time() - t0, 1), "n_new": len(new),
                  "text": tok.decode(new, skip_special_tokens=True)}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        log.info("generated row %s: %d tokens in %.0f s", r["row_id"], len(new), record["seconds"])
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    return 0


def _cos(a, b) -> float:
    import numpy as np

    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def cmd_recheck(args) -> int:
    """After a file is read: its first rows read again and compared with the saved reads (a redeploy check)."""
    import numpy as np

    setup_log(f"recheck_{args.name}")
    rows = load_rows(args.name)
    n_groups, n_points = shape_of(rows)
    saved = {}
    for f in sorted((ACTS / args.name).glob("*.npz")):
        with np.load(f) as z:
            saved.update({int(r): (z["content"][i], z["points"][i]) for i, r in enumerate(z["rows"])})
    pick = [r for r in rows if r["row_id"] in saved][: args.n]
    llm = load_remote_llm()
    content, points = with_watchdog(lambda: session_read(llm, [[r] for r in pick], int(llm.config.hidden_size), True,
                                                         MODEL["layers"], n_groups, n_points), 3600)
    c, p = masked(pick, content, points, n_groups, n_points)
    cos = []
    for i, r in enumerate(pick):
        for new, old in ((c[i], saved[r["row_id"]][0]), (p[i], saved[r["row_id"]][1])):
            for idx in np.ndindex(*new.shape[:2]):
                if np.isfinite(new[idx]).all() and np.isfinite(old[idx]).all():
                    cos.append(_cos(new[idx], old[idx]))
    result = {"rows_file": args.name, "rows": [r["row_id"] for r in pick], "min_cos": min(cos),
              "pass": min(cos) >= MIN_COS}
    (LOGS / f"recheck_{args.name}.json").write_text(json.dumps(result, indent=1))
    log.info("recheck %s: min cos %.7f; %s", args.name, min(cos), "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


def cmd_selftest(args) -> int:
    """Session output under remote="local" equals a plain HF forward: span means, points and log-probs, on a small
    random Llama with the real vocabulary."""
    import numpy as np
    import torch
    from nnsight import LanguageModel
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    setup_log("selftest")
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL["id"])
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=8,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=1024)
    hf = LlamaForCausalLM(cfg).eval()
    llm = LanguageModel(hf, tokenizer=tok)
    gen = torch.Generator().manual_seed(1)
    rows = []
    for i, (n, groups, points) in enumerate([
            (40, [[[2, 5], [10, 15]], [[20, 39]], []], [4, 14, -1]),
            (25, [[[5, 24]], [], [[1, 2]]], [24, -1, 0]),
            (60, [[[30, 40], [41, 59]], [[0, 1]], [[50, 59]]], [59, 30, 12])]):
        ids = torch.randint(1000, 120000, (n,), generator=gen).tolist()
        rows.append({"row_id": i, "ids": ids, "groups": groups, "points": points, "target_from": n - 7})
    layers = (1, 3, 6)  # hidden_states[-1] is the final norm's output, so the test reads blocks clear of the last
    traces = [[rows[0], rows[1]], [rows[2]]]
    content, points = session_read(llm, traces, 64, "local", layers, 3, 3)
    flat = [r for tr in traces for r in tr]
    c, p = masked(flat, content, points, 3, 3)
    lp = session_logprob(llm, rows, "local").numpy()
    worst, worst_lp, cases = 0.0, 0.0, 0
    with torch.no_grad():
        for i, r in enumerate(flat):
            hs = hf(torch.tensor([r["ids"]]), output_hidden_states=True).hidden_states
            for g, spans in enumerate(r["groups"]):
                idx = [t for a, b in spans for t in range(a, b)]
                for j, L in enumerate(layers):
                    if not idx:
                        assert np.isnan(c[i, g, j]).all(), "an empty group is not NaN"
                        continue
                    worst = max(worst, float(np.abs(c[i, g, j] - hs[L + 1][0, idx].mean(0).numpy()).max()))
                    cases += 1
            for q, pos in enumerate(r["points"]):
                for j, L in enumerate(layers):
                    if pos < 0:
                        assert np.isnan(p[i, q, j]).all(), "an absent point is not NaN"
                        continue
                    worst = max(worst, float(np.abs(p[i, q, j] - hs[L + 1][0, pos].numpy()).max()))
                    cases += 1
        for i, r in enumerate(rows):
            logits = hf(torch.tensor([r["ids"]])).logits[0].float().log_softmax(-1)
            ref = logits[r["target_from"] - 1:-1].gather(-1, torch.tensor(r["ids"][r["target_from"]:])[:, None])[:, 0]
            got = lp[i, : len(ref)]
            worst_lp = max(worst_lp, float(np.abs(got - ref.numpy()).max()))
            assert np.isnan(lp[i, len(ref):]).all()
    ok = worst < 1e-4 and worst_lp < 1e-4
    log.info("selftest: max |session - HF| = %.3g over %d comparisons; log-probs %.3g; %s", worst, cases, worst_lp,
             "PASS" if ok else "FAIL")
    (LOGS / "selftest.json").write_text(json.dumps({"max_abs_diff": worst, "max_abs_diff_logprob": worst_lp,
                                                    "comparisons": cases, "pass": ok}))
    return 0 if ok else 1


def main() -> int:
    setup_env()
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    for name in ("read", "logprob"):
        r = sub.add_parser(name)
        r.add_argument("--name", required=True)
        r.add_argument("--budget", type=int, default=7000)
        r.add_argument("--min_budget", type=int, default=2048)
        r.add_argument("--traces", type=int, default=8)
        r.add_argument("--session_timeout", type=float, default=2700.0)
        r.add_argument("--max_wait", type=float, default=4 * 3600.0)
        if name == "read":
            r.add_argument("--batched", action="store_true", help="pack rows into traces (default: one row per trace)")
            r.add_argument("--shard", default="0/1", help="K/N: only the rows at positions K mod N of the rows file")
    g = sub.add_parser("generate")
    g.add_argument("--name", required=True)
    g.add_argument("--retries", type=int, default=4)
    c = sub.add_parser("recheck")
    c.add_argument("--name", required=True)
    c.add_argument("--n", type=int, default=2)
    args = p.parse_args()
    return {"selftest": cmd_selftest, "read": cmd_read, "logprob": cmd_logprob, "generate": cmd_generate,
            "recheck": cmd_recheck}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
