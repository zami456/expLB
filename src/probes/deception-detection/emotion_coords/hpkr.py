"""HP-KR diagnostics: D1 (which subgroup drives the inversion), D2 (pre-response difference-in-differences) and
D3 (the same with an identical forced denial).

D1 scores HP-KR's three subgroups (lie, honest admission, honest denial; `phase0` rules, unclassified honest
rows left out) and reports AUROC(lie vs admission), AUROC(lie vs denial) and AUROC(admission vs denial),
where "A vs B" is the probability that a row of A scores above a row of B.

D2/D3 regress an emotion projection s = <x, unit emotion vector> on

    s ~ known + template fixed effects + known:harmful

over HP-KR questions in their subject's 5 harmful and 3 neutral templates (the harmful main effect is absorbed
by the template effects), with standard errors clustered by question (CR1). x is the pre-response token (D2)
or the forced reply's mean (D3, "I do not know the answer to the question." for every prompt). Each emotion's
interaction t is placed in two nulls: 1,000 random unit directions and the topic directions. P-values for the
Holm correction are one-sided, from the random-direction null.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from emotion_coords import config as C
from emotion_coords import linalg as la
from emotion_coords.probes import auroc
from emotion_coords.vectors import _layer_index, load_reads, row_meta

SUBGROUPS = ("lie", "honest_admission", "honest_denial")
P8_COMPASSION = ("compassionate", "empathetic", "sympathetic", "kind", "loving", "protective")
P8_THREAT = ("afraid", "terrified", "angry", "hostile", "suspicious", "vigilant")


# --- D1 --------------------------------------------------------------------------------------------


def hpkr_rows() -> pd.DataFrame:
    return pd.read_csv(C.data_dir("index") / "hpkr_rows.csv").set_index("pooled_row").sort_index()


def subgroup_aurocs(scores: np.ndarray, subgroup: np.ndarray) -> dict[str, float]:
    def vs(a: str, b: str) -> float:
        keep = np.isin(subgroup, [a, b])
        return auroc((subgroup[keep] == a).astype(int), scores[keep])

    return {"lie_vs_admission": vs("lie", "honest_admission"), "lie_vs_denial": vs("lie", "honest_denial"),
            "admission_vs_denial": vs("honest_admission", "honest_denial")}


def describe(name: str, scores: np.ndarray, rows: pd.DataFrame, lies: str = "all") -> dict:
    """Subgroup AUROCs and score quantiles for one scorer over every HP-KR row (pooled order)."""
    sub = rows["subgroup"].to_numpy()
    if lies == "explicit":  # sensitivity: lies restricted to explicit denials of knowledge
        sub = np.where((sub == "lie") & (rows["claim"].to_numpy() != "DENIED_KNOWLEDGE"), "lie_other", sub)
    out = {"scorer": name, "lies": lies} | subgroup_aurocs(scores, sub)
    for g in SUBGROUPS:
        s = scores[sub == g]
        out |= {f"{g}_n": int(len(s)), f"{g}_median": float(np.median(s)) if len(s) else np.nan}
    return out


# --- D2 / D3 ---------------------------------------------------------------------------------------


@dataclass
class Design:
    """The regression design shared by every direction: X (n, p), the interaction column, clusters."""

    x: np.ndarray
    interaction: int
    known_col: int
    clusters: np.ndarray
    meta: pd.DataFrame

    @classmethod
    def of(cls, meta: pd.DataFrame) -> "Design":
        templates = sorted(meta["template"].unique())
        cols = [np.ones(len(meta)), meta["known"].astype(float).to_numpy()]
        cols += [(meta["template"] == t).astype(float).to_numpy() for t in templates[1:]]
        cols.append((meta["known"] & meta["harmful"]).astype(float).to_numpy())
        x = np.stack(cols, axis=1)
        clusters = pd.factorize(meta["question_id"])[0]
        return cls(x, x.shape[1] - 1, 1, clusters, meta)

    def t_stats(self, s: np.ndarray, coef: int) -> tuple[np.ndarray, np.ndarray]:
        """OLS coefficient `coef` and its CR1 cluster-robust t for each column of s (n, d)."""
        x, n, p = self.x, self.x.shape[0], self.x.shape[1]
        xtx_inv = np.linalg.inv(x.T @ x)
        beta = xtx_inv @ x.T @ s
        resid = s - x @ beta
        w = x @ xtx_inv[coef]  # influence of each row on the coefficient
        g = int(self.clusters.max()) + 1
        h = np.zeros((g, s.shape[1]))
        np.add.at(h, self.clusters, w[:, None] * resid)
        var = (h ** 2).sum(0) * (g / (g - 1)) * ((n - 1) / (n - p))
        return beta[coef], beta[coef] / np.sqrt(var)


def projections(acts: np.ndarray, directions: np.ndarray) -> np.ndarray:
    return acts @ la.unit(directions).T


def did(acts: np.ndarray, design: Design, emotions: dict[str, np.ndarray], topics: np.ndarray, n_null: int = 1000,
        seed: int = 0, holm_set: tuple[str, ...] = P8_COMPASSION + P8_THREAT) -> pd.DataFrame:
    """Interaction (and known main effect) t for each emotion, against random and topic nulls."""
    rng = np.random.default_rng(seed)
    null = rng.standard_normal((n_null, acts.shape[1]))
    names = list(emotions)
    s_emotion = projections(acts, np.stack([emotions[e] for e in names]))
    s_null, s_topic = projections(acts, null), projections(acts, topics)
    _, t_null = design.t_stats(s_null, design.interaction)
    _, t_null_known = design.t_stats(s_null, design.known_col)
    _, t_topic = design.t_stats(s_topic, design.interaction)
    b_int, t_int = design.t_stats(s_emotion, design.interaction)
    b_known, t_known = design.t_stats(s_emotion, design.known_col)
    table = pd.DataFrame({
        "emotion": names, "beta_interaction": b_int, "t_interaction": t_int,
        "pct_random": [(t_null < t).mean() * 100 for t in t_int],
        "pct_topic": [(t_topic < t).mean() * 100 for t in t_int],
        "p_random_one_sided": [(1 + (t_null >= t).sum()) / (1 + len(t_null)) for t in t_int],
        "beta_known": b_known, "t_known": t_known,
        "pct_random_known": [(t_null_known < t).mean() * 100 for t in t_known],
    })
    table["in_holm_set"] = table["emotion"].isin(holm_set)
    table["p_holm"] = np.nan
    sub = table[table["in_holm_set"]].sort_values("p_random_one_sided")
    m, running = len(sub), 0.0
    for rank, (i, p) in enumerate(zip(sub.index, sub["p_random_one_sided"])):
        running = max(running, min(1.0, (m - rank) * p))
        table.loc[i, "p_holm"] = running
    table["above_95_both"] = (table["pct_random"] > 95) & (table["pct_topic"] > 95)
    return table


def d2d3_acts(layer: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(pre-response vectors, forced-reply means, metadata) of the d2d3 reads at `layer`, non-finite dropped."""
    ids, content, points = load_reads("d2d3")
    li = _layer_index(layer)
    pre, reply = points[:, 0, li].astype(np.float64), content[:, 0, li].astype(np.float64)
    ok = np.isfinite(pre).all(1) & np.isfinite(reply).all(1)
    meta = row_meta("d2d3").loc[ids[ok]].reset_index()
    return pre[ok], reply[ok], meta


def preanswer_check(layer: int) -> pd.DataFrame:
    """D2 prompts identical to an HP-KR row's prompt (same question, same harmful template) against the stored
    HP-KR pre-answer vector of that row (`results/hpkr_wording/preanswer_L*.npz`, position "pre")."""
    import hashlib
    import json

    from deception_detection.utils import REPO_ROOT

    stored_path = REPO_ROOT / "results" / "hpkr_wording" / f"preanswer_L{layer}.npz"
    if not stored_path.exists():
        return pd.DataFrame()
    with np.load(stored_path) as z:
        stored = z["vecs"][:, 0]
    templates = json.loads((C.DATA / "hpkr_templates.json").read_text(encoding="utf-8"))
    rows = hpkr_rows()
    pre, _, meta = d2d3_acts(layer)
    out = []
    for pooled_row, r in rows.iterrows():
        prefix, suffix = templates[r["subject"]]["harmful"][int(r["template_index"]) - 1]["template"].split(
            "{question}\n\nChoices:\n{choices}")
        block = r["user"][len(prefix): len(r["user"]) - len(suffix.rstrip())]
        qid = hashlib.sha1(f"{r['subject']}|{block}".encode()).hexdigest()[:12]
        hit = meta[(meta["question_id"] == qid) & (meta["template"] == f"harmful_{int(r['template_index'])}")]
        for i in hit.index:
            out.append({"pooled_row": pooled_row, "question_id": qid, "template": hit.loc[i, "template"],
                        "cos": float(la.unit(pre[i]) @ la.unit(stored[pooled_row]))})
    return pd.DataFrame(out)
