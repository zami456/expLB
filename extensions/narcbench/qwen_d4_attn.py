"""Post hoc (2026-10-24): an EleutherAI-style attention probe on D4's persona projection inside their pipeline
(Qwen3-32B, Verdict). Rows, basis and reader: `qwen_attn_rows.py`, `reader/seq_read.py` (declared before any read);
the probe: `attn_probe.py`; this file: declared before any score.

Per seed (their 10) and layer (26-30) and H in (1, 2, 8): their Core training rows, chosen as their
`train_deception_probe` chooses them (`their_selection`), each the persona-projected (k = 3) tokens of the reply span
of our Qwen3-32B read of the released Core transcripts; weight decay by 5-fold cross-validation over Core runs; the
fitted probe gives every Core and Verdict row its pooled vectors u = concat_h z_h, and their pipeline runs unchanged
on u, with their direction replaced by (identity scaler, v / |v|) -- then their score is the probe's output up to a
positive scale and a shift. Verdict spans: reflection = the reflection turn; decision = the verdict turn to the
decision token; pre_decision and verdict_start = the agent's last own turn before the verdict prompt (the same
vectors for both). Rose's label (Table 1) and Shi's.

Core held-out: our leave-one-domain-out (the probe refitted without the domain, their functions after), since their
`held_out_cv_core` keeps one set of vectors for every fold. References from the same reads and their unchanged
pipeline: their LR probe on the span's last token (their read position) and on the span mean. Null (H = 8, seed 0):
the 100 random 3-dim subspaces the reader built on the server (coordinates 3 + 3i ... 3 + 3i + 3), the probe
refitted in each with its own cross-validation. Probe fits run in `WORKERS` processes.
"""

import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from extensions.narcbench import attn_probe as AP
from extensions.narcbench import freeze as FZ
from extensions.narcbench import qwen_attn_rows as QA
from extensions.narcbench import qwen_d4_lb as QL
from extensions.narcbench import qwen_d4_pipeline as QP
from extensions.narcbench import spec as S
from extensions.narcbench import verdict_score as VS
from extensions.narcbench import verdict_spec as V

LAYERS = QP.LAYERS
HEADS = (1, 2, 8)
HEADLINE_H = 8
N_NULL = 100
NULL_OFFSET = 3  # null subspace i = read coordinates NULL_OFFSET + 3i ... + 3 (seq_read.py)
PERSONA = slice(0, 3)
WORKERS = 6
PHASE_SPAN = {"reflection": ("refl", 0), "decision": ("verdict", 1), "pre_decision": ("verdict", 0),
              "verdict_start": ("verdict", 0)}
LABELS = ("rose", "shi")
MARK = "== POST HOC ATTENTION-PROBE ANALYSIS"
OUT_JSON = S.RESULTS / "qwen_d4_attn.json"
OUT_MD = S.RESULTS / "qwen_d4_attn.md"
NULL_CKPT = V.ROOT / "attn_null_ckpt"


# ── data ─────────────────────────────────────────────────────────────────

def load_spans(kind: str, dims: slice, layer: int | None = None) -> list[list[np.ndarray]]:
    """Rows in file order -> spans -> (tokens, layers, d), or (tokens, d) at one layer index."""
    name = QA.NAMES[kind]
    order, lens = [], {}
    with open(V.ROWS / f"{name}.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            order.append(r["row_id"])
            lens[r["row_id"]] = [b - a for a, b in r["spans"]]
    got = {}
    for f in sorted((V.ACTS / "qwen" / name).glob("*.npz")):
        with np.load(f) as z:
            rows, pts = z["rows"], z["points"]
        for i, rid in enumerate(rows):
            sel = pts[i, :, :, :, dims] if layer is None else pts[i, :, :, layer, dims]
            got[int(rid)] = [np.ascontiguousarray(sel[g, :n]) for g, n in enumerate(lens[int(rid)])]
        del pts
    missing = [r for r in order if r not in got]
    assert not missing, f"{name}: {len(missing)} of {len(order)} rows not read"
    return [got[r] for r in order]


def _meta(kind: str) -> list[dict]:
    return json.loads((V.ROWS / f"{QA.NAMES[kind]}_meta.json").read_text(encoding="utf-8"))["rows"]


def verdict_order(phase: str) -> list[int]:
    """Index into our `kind` rows of each row of their phase metadata (their order)."""
    kind, _ = PHASE_SPAN[phase]
    pos = {(int(m["rep"]), int(m["episode_index"]), m["agent"]): j for j, m in enumerate(_meta(kind))}
    theirs = json.loads((VS.ROSE_RUN / "data" / "activations" / "qwen3_32b" / "verdict_decisions" /
                         f"metadata_{phase}.json").read_text(encoding="utf-8"))
    return [pos[(int(m["rep"]), int(m["episode_index"]), m["agent"])] for m in theirs]


def their_selection(meta: list[dict], seed: int, exclude_domain: str | None, mod) -> tuple[list[int], list[int]]:
    """The rows their `train_deception_probe` fits on, in its order: (honest, colluder)."""
    col, hon = [], []
    for i, m in enumerate(meta):
        if m["mode"] != "collusion" or m["channel"] != "public" or m["phase"] != "discussion":
            continue
        _, scenario = mod._parse_run_id(m["run_id"])
        if exclude_domain and mod._domain_from_scenario_id(scenario) == exclude_domain:
            continue
        if m["role"] == "colluder":
            col.append(i)
        elif m["role"] == "honest":
            hon.append(i)
    nc = min(len(col), len(hon))
    hon = list(np.random.RandomState(seed).choice(hon, nc, replace=False))
    return [int(h) for h in hon], col[:nc]


def selection_matches_theirs(mod, meta: list[dict]) -> bool:
    """Their `train_deception_probe` on row-index features, its LR's inputs captured: equal to `their_selection`."""
    seen = {}

    class Capture:
        def __init__(self, *a, **k):
            pass

        def fit(self, X, y):
            seen["X"], seen["y"] = X, y
            self.coef_ = np.ones((1, X.shape[1]))
            return self

    original = mod.LogisticRegression
    mod.LogisticRegression = Capture
    try:
        ok = True
        npz = {f"layer_{LAYERS[0]}": np.stack([np.arange(len(meta)), np.arange(len(meta)) ** 2], 1).astype(np.float64)}
        domains = sorted({mod._domain_from_scenario_id(mod._parse_run_id(m["run_id"])[1]) for m in meta})
        for seed in (0, 3):
            for dom in (None, domains[0]):
                scaler, _ = mod.train_deception_probe(meta, npz, LAYERS[0], exclude_domain=dom, seed=seed)
                rows = np.rint(scaler.inverse_transform(seen["X"])[:, 0]).astype(int).tolist()
                hon, col = their_selection(meta, seed, dom, mod)
                ok &= rows == hon + col and list(seen["y"]) == [0] * len(hon) + [1] * len(col)
        return bool(ok)
    finally:
        mod.LogisticRegression = original


# ── the probe inside their pipeline ──────────────────────────────────────

class Identity:
    def transform(self, X):
        return np.asarray(X, np.float64)


def fit_probe(core_spans, idx_hon, idx_col, groups, H, seed, wd=None):
    """(probe, mean, sd, wd, cv accuracies) on their training rows at one layer."""
    train = [core_spans[i] for i in idx_hon + idx_col]
    y = np.array([0] * len(idx_hon) + [1] * len(idx_col))
    mean, sd = AP.standardiser(train)
    b = AP.batch(train, mean, sd)
    acc = None
    if wd is None:
        wd, acc = AP.choose_weight_decay(b, y, [groups[i] for i in idx_hon + idx_col], H, seed)
    return AP.fit(b, y, H, wd, seed), mean, sd, wd, acc


def pooled(probe, spans, mean, sd, chunk: int = 256) -> np.ndarray:
    import torch

    out = []
    for s in range(0, len(spans), chunk):
        b = AP.batch(spans[s:s + chunk], mean, sd)
        with torch.no_grad():
            out.append(probe.pooled(b.x, b.mask, b.position).reshape(len(spans[s:s + chunk]), -1).numpy())
    return np.concatenate(out).astype(np.float64)


def entropy(probe, spans, mean, sd) -> float:
    """Mean over rows and heads of the attention entropy over log(span length) (spans of 2+ tokens)."""
    import torch

    keep = [s for s in spans if len(s) > 1]
    b = AP.batch(keep, mean, sd)
    with torch.no_grad():
        a = probe.attention(b.x, b.mask, b.position).numpy()
    n = b.mask.numpy().sum(1)
    ent = -(np.where(a > 0, a * np.log(np.clip(a, 1e-30, None)), 0.0)).sum(1) / np.log(n)[:, None]
    return float(ent.mean())


def their_raw(mods: dict, core_meta, core_u: dict, verdict_u: dict, fits: dict, seed: int) -> dict:
    """label -> their compute_all_results raw output, with `fits` (layer -> (scaler, w)) as their direction; their
    Core CV skipped (ours is separate)."""
    vmeta = {ph: mods["rose"].load_verdict_phase(ph)[0] for ph in V.QWEN_PHASES}
    out = {}
    for lab, mod in mods.items():
        mod.train_deception_probe = lambda meta, npz, layer, exclude_domain=None, seed=0: fits[layer]
        mod.held_out_cv_core = lambda *a, _m=mod, **k: {m: float("nan") for m in _m.METHOD_KEYS}
        vd = {ph: (vmeta[ph], {f"layer_{L}": verdict_u[ph][L] for L in LAYERS}) for ph in V.QWEN_PHASES}
        out[lab] = mod.compute_all_results(core_meta, {f"layer_{L}": core_u[L] for L in LAYERS}, vd, seed=seed)[1]
    return out


def _modules(labels: dict) -> dict:
    mods = {"rose": VS._their_module(VS.ROSE_RUN), "shi": VS._their_module(VS.ROSE_RUN)}
    QP._relabel(mods["shi"], labels["shi"])
    return mods


def _verdict_spans(spans: dict, phase: str, order: dict) -> list:
    kind, g = PHASE_SPAN[phase]
    return [spans[kind][j][g] for j in order[phase]]


def lodo(mod, core_meta, core_spans, groups, H, seed, wd, L) -> dict:
    """Their held_out_cv_core with the probe (and so the vectors) refitted per held-out domain."""
    base = mod.build_core_groups(core_meta, {f"layer_{L}": np.zeros((len(core_meta), 1))}, L)
    rids = sorted(base)
    domains = sorted({base[r]["domain"] for r in rids if base[r]["domain"]})
    res = {k: [] for k in mod.METHOD_KEYS}
    for dom in domains:
        tr = [r for r in rids if base[r]["domain"] != dom]
        te = [r for r in rids if base[r]["domain"] == dom]
        tr_y, te_y = mod._run_labels(base, tr), mod._run_labels(base, te)
        if len(np.unique(te_y)) < 2 or len(np.unique(tr_y)) < 2:
            continue
        hon, col = their_selection(core_meta, seed, dom, mod)
        probe, mean, sd, _, _ = fit_probe(core_spans, hon, col, groups, H, seed, wd=wd)
        u = pooled(probe, core_spans, mean, sd)
        g = mod.build_core_groups(core_meta, {f"layer_{L}": u}, L)
        w = AP.flat_direction(probe)
        w = w / np.linalg.norm(w)
        r = mod.evaluate(mod.compute_scores(g, tr, Identity(), w), tr_y, mod.compute_scores(g, te, Identity(), w), te_y,
                         seed=seed)
        if r:
            for k in mod.METHOD_KEYS:
                res[k].append(r[k])
    return {k: float(np.mean(v)) if v else float("nan") for k, v in res.items()}


SPAN_PHASES = ("reflection", "decision", "pre_decision")  # verdict_start reads pre_decision's span
_THEIRS = {}


def _their():
    if "mod" not in _THEIRS:
        _THEIRS["mod"] = VS._their_module(VS.ROSE_RUN)
    return _THEIRS["mod"]


def _job(p: dict) -> dict:
    """One probe fit, in a worker: their training rows at one layer -> its direction, every row's pooled vectors,
    and the extras the payload asks for (leave-one-domain-out, the identity check and attention entropy)."""
    import torch

    torch.set_num_threads(2)
    probe, mean, sd, wd, acc = fit_probe(p["core"], p["hon"], p["col"], p["groups"], p["H"], p["seed"], wd=p.get("wd"))
    w = AP.flat_direction(probe)
    out = {"w": w / np.linalg.norm(w), "wd": wd, "acc": acc, "core_u": pooled(probe, p["core"], mean, sd),
           "verdict_u": {ph: pooled(probe, s, mean, sd) for ph, s in p["verdict"].items()}}
    if p.get("extras"):
        b = AP.batch(p["core"][:64], mean, sd)
        lin = out["core_u"][:64] @ w + float(probe.v.bias.detach().sum())
        out["identity_err"] = float(np.abs(lin - AP.scores(probe, b)).max())
        out["entropy"] = {ph: entropy(probe, s, mean, sd) for ph, s in p["verdict"].items()} | {
            "core": entropy(probe, p["core"], mean, sd)}
    if p.get("lodo"):
        out["lodo"] = lodo(_their(), p["meta"], p["core"], p["groups"], p["H"], p["seed"], wd, p["L"])
    return out


def _pool(payloads: list[dict]) -> list[dict]:
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(_job, payloads))


def _assemble(results: dict, seed_layer_keys) -> tuple[dict, dict, dict]:
    """(fits, core_u, verdict_u) over layers, verdict_start given pre_decision's vectors."""
    fits, core_u, verdict_u = {}, {}, {ph: {} for ph in V.QWEN_PHASES}
    for L, res in seed_layer_keys(results):
        fits[L], core_u[L] = (Identity(), res["w"]), res["core_u"]
        for ph in SPAN_PHASES:
            verdict_u[ph][L] = res["verdict_u"][ph]
        verdict_u["verdict_start"][L] = res["verdict_u"]["pre_decision"]
    return fits, core_u, verdict_u


def run() -> None:
    _check_declared()
    VS.check_freeze()
    os.environ["MPLBACKEND"] = "Agg"
    labels = VS.load_labels(V.ROWS / "labels.json")
    core_meta = _meta("core")
    groups = [m["run_id"] for m in core_meta]
    probe_mod = VS._their_module(VS.ROSE_RUN)
    order = {ph: verdict_order(ph) for ph in V.QWEN_PHASES}
    spans = {kind: load_spans(kind, PERSONA) for kind in QA.NAMES}
    core = [r[0] for r in spans["core"]]
    vsp = {ph: _verdict_spans(spans, ph, order) for ph in SPAN_PHASES}
    out = {"note": "POST HOC, descriptive: attention probe (EleutherAI) on D4's persona projection (k = 3) inside "
                   "their pipeline; Qwen3-32B reads", "checks": {
                       "selection_matches_theirs": selection_matches_theirs(probe_mod, core_meta)},
           "rows": {k: len(v) for k, v in spans.items()}}
    assert out["checks"]["selection_matches_theirs"], "their_selection differs from their train_deception_probe"
    keys, payloads = [], []
    for seed in range(probe_mod.N_SEEDS):
        hon, col = their_selection(core_meta, seed, None, probe_mod)
        for H in HEADS:
            for i, L in enumerate(LAYERS):
                keys.append((seed, H, L))
                payloads.append({"core": [s[:, i] for s in core], "verdict": {ph: [s[:, i] for s in vsp[ph]] for ph in vsp},
                                 "hon": hon, "col": col, "groups": groups, "meta": core_meta, "H": H, "seed": seed,
                                 "L": L, "lodo": True, "extras": seed == 0 and H == HEADLINE_H})
    print(f"{datetime.now():%H:%M:%S} {len(payloads)} probe fits on {WORKERS} workers", flush=True)
    res = dict(zip(keys, _pool(payloads)))
    del payloads
    raws = {H: {lab: [] for lab in LABELS} for H in HEADS}
    held = {H: [] for H in HEADS}
    for seed in range(probe_mod.N_SEEDS):
        for H in HEADS:
            fits, core_u, verdict_u = _assemble(res, lambda r, s=seed, h=H: [(L, r[(s, h, L)]) for L in LAYERS])
            r = their_raw(_modules(labels), core_meta, core_u, verdict_u, fits, seed)
            for lab in LABELS:
                raws[H][lab].append(r[lab])
            held[H].append({L: res[(seed, H, L)]["lodo"] for L in LAYERS})
            print(f"{datetime.now():%H:%M:%S} seed {seed} H {H}: reflection rose "
                  + " ".join(f"{np.mean(r['rose']['reflection'][m]):.3f}" for m in QP.METHODS), flush=True)
    wd_chosen = {f"H{H}|s{s}|L{L}": {"wd": v["wd"], "cv_accuracy": v["acc"]} for (s, H, L), v in res.items()}
    ent = {f"L{L}": res[(0, HEADLINE_H, L)]["entropy"] for L in LAYERS}
    identity_err = max(res[(0, HEADLINE_H, L)]["identity_err"] for L in LAYERS)
    _write(out, raws, held, wd_chosen, ent, identity_err)
    out["references"] = references(labels, spans, order, core_meta)
    _write(out, raws, held, wd_chosen, ent, identity_err)
    out["null"] = null(labels, core_meta, groups, order, raws[HEADLINE_H])
    _write(out, raws, held, wd_chosen, ent, identity_err)


def _write(out, raws, held, wd_chosen, ent, identity_err) -> None:
    groups = list(V.QWEN_PHASES)
    out = out | {"attn": {str(H): {lab: QK_stat(raws[H][lab], groups) for lab in LABELS} for H in HEADS},
                 "core_lodo": {str(H): {m: _ms([x[L][m] for x in held[H] for L in LAYERS]) for m in QP.METHODS}
                               for H in HEADS},
                 "weight_decay": wd_chosen, "entropy_H8_seed0": ent}
    out["checks"] = out["checks"] | {"linear_identity_max_err": identity_err}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def QK_stat(raws: list[dict], groups) -> dict:
    from extensions.narcbench import qwen_d4_ksweep as QK

    return QK.stat(raws, groups)


def _ms(vals) -> list[float]:
    a = np.array(vals, float)
    n = int(np.sum(~np.isnan(a)))
    return [float(np.nanmean(a)), float(np.nanstd(a) / np.sqrt(n))]


def references(labels, spans, order, core_meta) -> dict:
    """Their pipeline unchanged (their probe, their Core CV) on the k = 3 span's last token and on the span mean."""
    out = {}
    for name, agg in (("span_last_token", lambda s: s[-1]), ("span_mean", lambda s: s.mean(0))):
        core_npz = {f"layer_{L}": np.stack([agg(r[0][:, i]) for r in spans["core"]]) for i, L in enumerate(LAYERS)}
        res = {}
        for lab, mod in _modules(labels).items():
            vd = {ph: (mod.load_verdict_phase(ph)[0], {f"layer_{L}": np.stack(
                [agg(s[:, i]) for s in _verdict_spans(spans, ph, order)]) for i, L in enumerate(LAYERS)})
                  for ph in V.QWEN_PHASES}
            res[lab] = QK_stat([mod.compute_all_results(core_meta, core_npz, vd, seed=s)[1] for s in range(mod.N_SEEDS)],
                               ["Held-out"] + list(V.QWEN_PHASES))
        out[name] = res
    return out


def null(labels, core_meta, groups, order, persona_raws: dict) -> dict:
    """H = 8, seed 0: the probe in each of the 100 server-built random 3-dim subspaces (own cross-validation)."""
    probe_mod = VS._their_module(VS.ROSE_RUN)
    hon, col = their_selection(core_meta, 0, None, probe_mod)
    res = {}
    for i, L in enumerate(LAYERS):
        ckpt = NULL_CKPT / f"L{L}.npz"
        if ckpt.exists():  # a layer fitted before a stop (each layer's fits are saved as soon as they finish)
            with np.load(ckpt) as z:
                for j in range(N_NULL):
                    res[(j, L)] = {"w": z["w"][j], "core_u": z["core_u"][j],
                                   "verdict_u": {ph: z[f"v_{ph}"][j] for ph in SPAN_PHASES}}
            print(f"{datetime.now():%H:%M:%S} null layer {L} loaded from {ckpt.name}", flush=True)
            continue
        sp = {kind: load_spans(kind, slice(NULL_OFFSET, NULL_OFFSET + 3 * N_NULL), layer=i) for kind in QA.NAMES}
        vsp = {ph: _verdict_spans(sp, ph, order) for ph in SPAN_PHASES}
        payloads = [{"core": [r[0][:, 3 * j:3 * j + 3] for r in sp["core"]],
                     "verdict": {ph: [s[:, 3 * j:3 * j + 3] for s in vsp[ph]] for ph in vsp},
                     "hon": hon, "col": col, "groups": groups, "H": HEADLINE_H, "seed": 0, "L": L}
                    for j in range(N_NULL)]
        for j, r in enumerate(_pool(payloads)):
            res[(j, L)] = r
        del sp, vsp, payloads
        NULL_CKPT.mkdir(parents=True, exist_ok=True)
        part = ckpt.with_suffix(".partial")
        with open(part, "wb") as f:
            np.savez(f, w=np.stack([res[(j, L)]["w"] for j in range(N_NULL)]),
                     core_u=np.stack([res[(j, L)]["core_u"] for j in range(N_NULL)]),
                     **{f"v_{ph}": np.stack([res[(j, L)]["verdict_u"][ph] for j in range(N_NULL)]) for ph in SPAN_PHASES})
        os.replace(part, ckpt)
        print(f"{datetime.now():%H:%M:%S} null layer {L} fitted", flush=True)
    nulls = []
    for j in range(N_NULL):
        fits, core_u, verdict_u = _assemble(res, lambda r, j=j: [(L, r[(j, L)]) for L in LAYERS])
        nulls.append(their_raw(_modules(labels), core_meta, core_u, verdict_u, fits, 0))
    band = {}
    for lab in LABELS:
        for ph in V.QWEN_PHASES:
            for m in QP.METHODS:
                obs = float(np.mean(persona_raws[lab][0][ph][m]))
                vals = [float(np.mean(x[lab][ph][m])) for x in nulls]
                band.setdefault(lab, {}).setdefault(ph, {})[m] = {
                    "observed_seed0": obs, "band97.5": float(np.percentile(vals, 97.5)), "median": float(np.median(vals)),
                    "p": float((1 + sum(v >= obs for v in vals)) / (1 + len(vals)))}
    return {"n": N_NULL, "heads": HEADLINE_H, "note": "seed 0; mean over layers 26-30; 100 random orthonormal 3-dim "
                                                       "subspaces built on the server (seq_read.null_basis); own "
                                                       "weight-decay CV", "bands": band}


def null_only() -> None:
    """The null alone, after `run` saved everything else: a power loss (2026-10-24 12:10) stopped `run` in its null.
    The seed-0, H = 8 persona fits it compares with are redone exactly as `run` did them (their chosen weight decays
    and cross-validation accuracies checked against the saved ones), then the null, checkpointed per layer."""
    _check_declared()
    VS.check_freeze()
    os.environ["MPLBACKEND"] = "Agg"
    out = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    labels = VS.load_labels(V.ROWS / "labels.json")
    core_meta = _meta("core")
    groups = [m["run_id"] for m in core_meta]
    probe_mod = VS._their_module(VS.ROSE_RUN)
    order = {ph: verdict_order(ph) for ph in V.QWEN_PHASES}
    spans = {kind: load_spans(kind, PERSONA) for kind in QA.NAMES}
    core = [r[0] for r in spans["core"]]
    vsp = {ph: _verdict_spans(spans, ph, order) for ph in SPAN_PHASES}
    hon, col = their_selection(core_meta, 0, None, probe_mod)
    payloads = [{"core": [s[:, i] for s in core], "verdict": {ph: [s[:, i] for s in vsp[ph]] for ph in vsp},
                 "hon": hon, "col": col, "groups": groups, "H": HEADLINE_H, "seed": 0, "L": L}
                for i, L in enumerate(LAYERS)]
    res = dict(zip(LAYERS, _pool(payloads)))
    saved = out["weight_decay"]
    same = all(res[L]["wd"] == saved[f"H{HEADLINE_H}|s0|L{L}"]["wd"]
               and {str(k): v for k, v in res[L]["acc"].items()} == saved[f"H{HEADLINE_H}|s0|L{L}"]["cv_accuracy"]
               for L in LAYERS)
    print(f"{datetime.now():%H:%M:%S} seed-0 H = {HEADLINE_H} persona fits redone; same as run's: {same}", flush=True)
    assert same, "the redone seed-0 persona fits differ from run's"
    fits, core_u, verdict_u = _assemble(res, lambda r: [(L, r[L]) for L in LAYERS])
    persona = their_raw(_modules(labels), core_meta, core_u, verdict_u, fits, 0)
    out["null"] = null(labels, core_meta, groups, order, {lab: [persona[lab]] for lab in LABELS})
    out["checks"] = out["checks"] | {"null_seed0_refit_matches_run": same}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


# ── freeze ───────────────────────────────────────────────────────────────

def declare() -> None:
    lines = [f"{MARK} (declared before any score)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_d4_attn.py sha256 {FZ.sha256(Path(__file__))}",
             f"attn_probe.py sha256 {FZ.sha256(Path(AP.__file__))}",
             "probe as EleutherAI's code (attention_probe.py, trainer.py, train_mosaic.py, read 2026-10-24): q zero-init "
             "no bias, v with bias, per-head position weight x position from the span's first token, -1e9 mask, "
             "softmax over the span, heads summed; BCE + (wd/2)|theta|^2, full batch, LBFGS strong Wolfe max_iter "
             "2000 (their default; the plan's AdamW 500 steps replaced: their AdamW default lr 1e-4 cannot move the "
             "zero-initialised query in 500 steps), retrain while loss > 0.5 up to 5 inits (lowest loss kept); wd "
             "from {0, 1e-3, 1e-2, 1e-1} by 5-fold CV (StratifiedGroupKFold by run, seed), highest mean accuracy "
             "(logit > 0), ties to the smaller, per (seed, layer, H); inputs standardised over the training tokens",
             f"H {HEADS} (headline {HEADLINE_H}), seeds 0-9, layers {LAYERS}, k = 3 (basis rows 0-2); training rows = "
             "their train_deception_probe's selection (checked against it at run time); null H = 8, seed 0, the "
             "100 server-built random subspaces (read coordinates 3-302), own CV; Core held-out = our "
             "leave-one-domain-out (weight decay as chosen on all of Core) with the probe "
             "refitted per domain and their functions after; references: their pipeline on the span's last token and "
             "on the span mean; descriptive, no Holm", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def declare_edit() -> None:
    """A post-freeze edit (mechanics only): the block restates both hashes under MARK, so `_check_declared` (the
    last block under MARK) takes it."""
    lines = [f"{MARK}, POST-FREEZE EDIT (mechanics only; no score of the null exists)  {datetime.now():%Y-%m-%d %H:%M:%S}",
             f"qwen_d4_attn.py sha256 {FZ.sha256(Path(__file__))}",
             f"attn_probe.py sha256 {FZ.sha256(Path(AP.__file__))} (unchanged)",
             "a power loss at 12:10 (Kernel-Power 41, reboot 13:51) stopped run in its null (layer 29; layers 26-28 "
             "were fitted in memory and lost) after the main fits and references were saved (07:31); added: per-layer "
             "null checkpoints (E:/narcbench/verdict/attn_null_ckpt) and null_only, which redoes the seed-0 H = 8 "
             "persona fits the null compares with (checked equal to run's saved weight decays and CV accuracies), then "
             "the null as declared; the report notes that check; nothing computed changes", ""]
    with open(S.LOGS / "freeze.txt", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def _check_declared() -> None:
    text = (S.LOGS / "freeze.txt").read_text(encoding="utf-8-sig")
    rows = QL._shas(QL._block(text, QA.MARK))
    mine = QL._shas(QL._block(text, MARK))
    want_rows = {f.name: FZ.sha256(f) for f in [V.ROWS / f"{n}{s}" for n in QA.NAMES.values()
                                                for s in (".jsonl", "_meta.json")]}
    for k, v in want_rows.items():
        if rows.get(k) != v:
            raise SystemExit(f"{k} changed since its declaration")
    if rows.get(QA.BASIS_FILE.name) != FZ.sha256(QA.BASIS_FILE):
        raise SystemExit("the basis changed since its declaration")
    if mine != {"qwen_d4_attn.py": FZ.sha256(Path(__file__)), "attn_probe.py": FZ.sha256(Path(AP.__file__))}:
        raise SystemExit("qwen_d4_attn.py or attn_probe.py changed since their declaration")


# ── report ───────────────────────────────────────────────────────────────

def report() -> None:
    r = json.loads(OUT_JSON.read_text(encoding="utf-8"))
    prev = json.loads(QP.OUT_JSON.read_text(encoding="utf-8"))
    N, ms = QP.NAMES, QP._ms
    nb = r["null"]["bands"]
    ref = r["references"]
    heads = " | ".join(f"Attention H={H}" for H in HEADS)
    band = f"Random 3-dim, H={HEADLINE_H}: band (p)"
    out = ["# Attention probe on D4's persona projection inside their pipeline (Qwen3-32B, Verdict)", "",
           "Post hoc and descriptive. An EleutherAI-style attention probe (their code's architecture and training: "
           "zero-initialised queries, a learned position weight per head, heads summed, LBFGS, weight decay by 5-fold "
           "cross-validation) reads the k = 3 persona-projected tokens of each scored turn, read by Qwen3-32B with the "
           "full context. It is trained on their Core training rows (colluder vs honest, their selection) from our "
           "read of the released Core transcripts, and enters their pipeline unchanged through its pooled vectors. "
           "± is their standard error over 10 seeds x 5 layers. Rows, basis, reader and this analysis were declared "
           "before any read or score (`E:/narcbench/logs/freeze.txt`).", "",
           "## Reflection phase, Rose's label (their Table 1's setting)", "",
           "| Method | Table 1 | Unprojected | Projected, last token (their Core) | Span last token | Span mean | "
           f"{heads} | {band} |", "|---|" + "---|" * (6 + len(HEADS))]
    for m in QP.METHODS:
        b = nb["rose"]["reflection"][m]
        out.append(f"| {N[m]} | {ms(QP.TABLE1[m])} | {ms(prev['unprojected_summary_rose']['reflection'][m])} | "
                   f"{ms(prev['projected_summary_rose']['reflection'][m])} | "
                   f"{ms(ref['span_last_token']['rose']['reflection'][m])} | {ms(ref['span_mean']['rose']['reflection'][m])} | "
                   + " | ".join(ms(r["attn"][str(H)]["rose"]["reflection"][m]) for H in HEADS)
                   + f" | {b['band97.5']:.3f} ({b['p']:.2f}) |")
    for ph, lab, title in (("reflection", "shi", "Reflection phase, Shi's label"),
                           ("pre_decision", "shi", "Before the verdict (the agent's last own turn), Shi's label"),
                           ("pre_decision", "rose", "Before the verdict (the agent's last own turn), Rose's label"),
                           ("decision", "rose", "Decision token (the verdict turn), Rose's label")):
        out += ["", f"## {title}", "", f"| Method | Span last token | Span mean | {heads} | {band} |",
                "|---|" + "---|" * (3 + len(HEADS))]
        for m in QP.METHODS:
            b = nb[lab][ph][m]
            out.append(f"| {N[m]} | {ms(ref['span_last_token'][lab][ph][m])} | {ms(ref['span_mean'][lab][ph][m])} | "
                       + " | ".join(ms(r["attn"][str(H)][lab][ph][m]) for H in HEADS)
                       + f" | {b['band97.5']:.3f} ({b['p']:.2f}) |")
    out += ["", "pre_decision and verdict_start use the same span (the agent's last own turn), so verdict_start is in "
            "the JSON only. p = (1 + random subspaces at or above) / 101 (seed 0, layer mean); unadjusted, post hoc.", "",
            "## Core held-out domain (leave one domain out, probe refitted per domain)", "",
            f"| Method | Span last token (their CV) | Span mean (their CV) | {heads} |", "|---|" + "---|" * (2 + len(HEADS))]
    for m in QP.METHODS:
        out.append(f"| {N[m]} | {ms(ref['span_last_token']['rose']['Held-out'][m])} | {ms(ref['span_mean']['rose']['Held-out'][m])} | "
                   + " | ".join(ms(r["core_lodo"][str(H)][m]) for H in HEADS) + " |")
    ent = r["entropy_H8_seed0"]
    keys = list(next(iter(ent.values())))
    out += ["", "## Attention entropy (H = 8, seed 0; 1 = uniform over the span)", "",
            "| Layer | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for L, row in ent.items():
        out.append(f"| {L} | " + " | ".join(f"{row[k]:.2f}" for k in keys) + " |")
    wds = [v["wd"] for v in r["weight_decay"].values()]
    out += ["", f"Weight decay chosen: " + ", ".join(f"{w:g} x{wds.count(w)}" for w in sorted(set(wds))) + ".",
            f"Checks: their training rows reproduced {r['checks']['selection_matches_theirs']}; pooled-vector identity "
            f"max error {r['checks']['linear_identity_max_err']:.1e}"
            + (f"; the null was rerun after a power loss, its seed-0 persona fits equal run's "
               f"{r['checks']['null_seed0_refit_matches_run']}" if "null_seed0_refit_matches_run" in r["checks"] else "")
            + ".", ""]
    OUT_MD.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {OUT_MD} ({len(out)} lines)")


if __name__ == "__main__":
    fire.Fire({"declare": declare, "declare_edit": declare_edit, "run": run, "null_only": null_only, "report": report})
