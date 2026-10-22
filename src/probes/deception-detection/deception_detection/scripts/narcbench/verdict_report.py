"""Verdict report: `results/narcbench/verdict_summary.md`, built only from `verdict_llama.json`, `verdict_qwen.json`
and the gate logs; sections whose inputs do not exist are skipped, and two builds give the same file."""

import json
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import spec as S
from deception_detection.scripts.narcbench import verdict_spec as V

METHODS = ("peak_suspicion", "deception_split", "activation_alignment", "asymmetry_probe", "sorted_concat")
SHORT = {"peak_suspicion": "Peak", "deception_split": "Split", "activation_alignment": "Align",
         "asymmetry_probe": "Asym", "sorted_concat": "Concat"}
DETECTORS = ("D9+K", "D4 (LB)", "rose_exact", "D9", "K", "D5+K", "RepE", "D4 (Core)", "rose_L40", "rose_converged")
NAMES = {"rose_exact": "their probe", "rose_L40": "their recipe, L40 span", "rose_converged": "their probe, converged"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def _ci(c):
    return f"[{_f(c[0])}, {_f(c[1])}]" if c else ""


def llama_section(r: dict) -> list[str]:
    out = ["## Llama-3.3-70B arm (frozen detectors; nothing fitted on Verdict)", "",
           f"{r['episodes']} episodes, {r['collusion']} colluding by Shi's label (both agents ACCEPT under the raw-log "
           f"rule); {r['labels']['rose_episodes']} by Rose's label; explicit agreement in "
           f"{r['labels']['explicit_agreement']}. Episode score: the higher of the two agents' scores (Peak Suspicion).", "",
           "### Primary tests (Holm over six, alpha 0.05)", "",
           "| Test | Detector | Readout | AUROC [95% CI over runs] | Random band | p | Holm |", "|---|---|---|---|---|---|---|"]
    for test, t in r["primary"].items():
        h = r["holm"][test]
        out.append(f"| {test} | {NAMES.get(t['detector'], t['detector'])} | {t['readout']} | {_f(t['auroc'], 3)} "
                   f"{_ci(t['ci'])} | {_f(t['band97.5'])} | {t['p']:.4f} | "
                   f"{'reject' if h['reject'] else 'not rejected'} (threshold {h['holm_threshold']:.4f}) |")
    out += ["", "COMM: the agent's own messages before its verdict; REFL: its private reflection after the outcome. "
            "Random band: 97.5th percentile of 1,000 frozen random directions read and aggregated the same way.", "",
            "Estimation (ΔAUROC, 95% CI over runs): " + "; ".join(
                f"{k} {_f(v['point'], 3)} {_ci(v['ci'])}" for k, v in r["estimation"].items()) + ".", "",
            "### Every detector and readout (AUROC against Shi's label; Rose's label in brackets)", "",
            "| Detector | " + " | ".join(V_READOUTS) + " |", "|---|" + "---|" * len(V_READOUTS)]
    for det in DETECTORS:
        cells = []
        for ro in V_READOUTS:
            e = r["readouts"].get(ro, {}).get(det)
            cells.append(f"{_f(e['auroc_shi'])} ({_f(e['auroc_rose'])})" if e else "–")
        out.append(f"| {NAMES.get(det, det)} | " + " | ".join(cells) + " |")
    out += ["", "VERDICT holds the verdict itself and PRE is the last token before it; BARE_* read the text after a "
            "neutral prompt. The VERDICT readout contains the label (a leak check, not a test).", "",
            "Random bands by readout: " + "; ".join(
                f"{ro} thesis {_f(n.get('random', {}).get('band97.5'))}, their space {_f(n.get('rose_random', {}).get('band97.5'))}"
                for ro, n in r["nulls"].items()) + ".", "",
            "Random persona splits (99): " + "; ".join(
                f"{ro} {k} split0 {_f(n[k]['split0'])}, null median {_f(n[k]['median'])}, p {_f(n[k]['p'], 3)}"
                for ro, n in r["nulls"].items() for k in ("R9", "R9+K") if k in n) + ".", "",
            "Agent level, REFL against Shi's relaxation judge (policy relaxation in the private reflection, "
            f"{r['labels']['relaxation_true']} of 1,000 positive): " + ", ".join(
                f"{NAMES.get(d, d)} {_f(r['readouts']['REFL'][d].get('relaxation_agent_auroc'))}"
                for d in ("D9+K", "D4 (LB)", "rose_exact") if d in r["readouts"].get("REFL", {})) + ".", "",
            "AUROC by episode (1-10), Shi's label: " + "; ".join(
                f"{NAMES.get(d, d)} {ro} " + " ".join(_f(x) for x in r["readouts"][ro][d]["by_episode"])
                for d in ("D9+K", "D4 (LB)", "rose_exact") for ro in ("COMM", "REFL") if d in r["readouts"].get(ro, {})) + "."]
    tb = r.get("text_baselines") or {}
    if tb:
        out += ["", "Text baselines (episode level; Shi's label, Rose's in brackets): " + "; ".join(
            f"{span}: 'accept' present {_f(v['shi']['keyword_accept'])} ({_f(v['rose']['keyword_accept'])}), TF-IDF + "
            f"logistic regression cross-validated by run {_f(v['shi']['tfidf_cv_by_run'])} ({_f(v['rose']['tfidf_cv_by_run'])})"
            for span, v in tb.items()) + "."]
    return out + [""]


V_READOUTS = ("COMM", "PRE", "REFL", "VERDICT", "BARE_COMM", "BARE_REFL")


def qwen_section(q: dict) -> list[str]:
    phases = list(V.QWEN_PHASES)
    a, b = q["their_script_rose_label"]["mean"], q["their_functions_shi_label"]
    out = ["## Qwen3-32B arm (their pipeline: their Core activations, their script)", "",
           f"{q['rows']} agent-episodes read on NDIF (bf16) at their four positions, layers 26-30.", "",
           "| Label | Phase | " + " | ".join(SHORT[m] for m in METHODS) + " |", "|---|---|" + "---|" * len(METHODS)]
    for lab, src in (("Rose (their script, unchanged)", a), ("Shi (their functions)", b)):
        for ph in ["Held-out"] + phases:
            if ph in src:
                out.append(f"| {lab} | {'Core held-out domain' if ph == 'Held-out' else ph} | "
                           + " | ".join(_f(src[ph].get(m)) for m in METHODS) + " |")
    out.append("| Paper Table 1 | reflection | " + " | ".join(_f(V.PAPER_TABLE1_REFLECTION[m]) for m in METHODS) + " |")
    band = q.get("random_band_seed0") or {}
    if band:
        out += ["", "Random directions in their space (97.5th percentile of the layer mean, seed-0 scaler): " + "; ".join(
            f"{lab} {ph} " + ", ".join(f"{SHORT[m]} {_f(v[m]['band97.5'])}" for m in v)
            for lab, phs in band.items() for ph, v in phs.items()) + "."]
    awq = q.get("awq_vs_bf16")
    if awq:
        out += ["", "AWQ (their Core) vs bf16 (our reads), 20 released Transfer rows: " + "; ".join(
            f"{L} score r {_f(v['score_pearson_r'], 3)}, cos median {_f(v['cos_median'], 4)}" for L, v in awq.items()) + "."]
    out += ["", "The `decision` position is the accept/reject token itself and so carries Shi's label (both ACCEPT)."]
    return out + [""]


def caveats() -> list[str]:
    return ["## Design, limitations and disclosure", "",
            f"- Data: Shi et al. (arXiv:2609.24967), `{V.HF_DATASET}` revision `{V.HF_REVISION[:8]}`, condition "
            f"`{V.CONDITION}` (50 runs x 10 episodes x 2 agents); their code `{V.SHI_COMMIT[:8]}`; Rose et al.'s code "
            f"`{S.GITHUB_COMMIT[:8]}`. No agent is instructed to collude.",
            "- Compact context (pre-registered): the system turn, each earlier episode's outcome feedback, reflection prompt "
            "and reflection (Shi's feedback-and-reflection memory), and the current episode with its task-phase tool outputs "
            "replaced by a length note; the agents saw the full history (17k-67k tokens), which NDIF cannot read.",
            "- Reads are off-policy: Gemma-4-31B wrote every turn; Llama-3.3-70B and Qwen3-32B read them. Each row is read "
            "alone (batching moves single-token vectors slightly; solo reads reproduce exactly).",
            "- Collusion here is tacit (explicit agreement in 2 of 146 colluding episodes); Shi's label is mutual ACCEPT, so "
            "readouts that contain the verdict (VERDICT, `decision`) carry the label, and REFL follows the outcome feedback.",
            "- Their Core probe is trained on AWQ activations and applied to bf16 reads (the AWQ check above).",
            "- Exposure before freeze V: rep001's structure and token counts (Gemma; Qwen3.8 in another condition), one "
            "verdict submission, Alice's turn list for episodes 1 and 4, and the label base rates; no Verdict score existed.",
            "- Sprint work (Apart AI Collusion sprint, Oct 23-25); the NARCBench audit before it is disclosed as prior work.", ""]


def build() -> None:
    r, q = _load(V.RESULTS / "verdict_llama.json"), _load(V.RESULTS / "verdict_qwen.json")
    lines = ["# Verdict: the collusion detectors on emergent collusion (Shi et al.'s runs, NDIF reads)", "",
             "Frozen before any read (freeze V in `E:/narcbench/logs/freeze.txt`); rules in `verdict_spec.py`; gates in "
             "`E:/narcbench/logs/gates.md`.", ""]
    if r:
        lines += llama_section(r)
    if q:
        lines += qwen_section(q)
    lines += caveats()
    (V.RESULTS / "verdict_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {V.RESULTS / 'verdict_summary.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    fire.Fire({"build": build})
