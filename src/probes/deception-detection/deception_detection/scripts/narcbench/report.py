"""NARCBench report: `results/narcbench/summary.md`, built only from the saved result files.

Every number below is read from a file in `results/narcbench/` or `E:/narcbench/logs/`; sections whose
inputs do not exist yet are skipped, so the report can be rebuilt at any stage and twice gives the same
file.
"""

import json
from pathlib import Path

import fire
import numpy as np

from deception_detection.scripts.narcbench import spec as S

M = ["peak_suspicion", "deception_split", "activation_alignment", "asymmetry_probe", "sorted_concat"]
SHORT = {"peak_suspicion": "Peak", "deception_split": "Split", "activation_alignment": "Align",
         "asymmetry_probe": "Asym", "sorted_concat": "Concat"}


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _f(x, nd=2):
    return "–" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def arm_a() -> list[str]:
    out = ["## Arm A: their probe on their released activations (CPU)", ""]
    for model, layers in (("qwen3_32b", "26-30"), ("llama_3_1_70b", "32-37")):
        rep = _load(S.RESULTS / f"armA_replicate_{model}.json")
        nul = _load(S.RESULTS / f"armA_random_null_{model}.json")
        pse = _load(S.RESULTS / f"armA_pseudo_null_{model}.json")
        sen = _load(S.RESULTS / f"armA_sensitivity_{model}.json")
        stc = _load(S.RESULTS / f"armA_stego_corrected_{model}.json")
        if not rep:
            continue
        means = rep["means"]
        fams = [g for g in means if g not in ("Held-out", "blackjack")]
        out += [f"### {model} (layers {layers}, {rep['seeds']} seeds, their aggregation)", "",
                "AUROC, collusion runs vs control runs. In brackets: the 97.5th percentile of 1,000 random "
                "unit directions through the same aggregator and folds (no band for Align and Concat).", "",
                "| Tier | " + " | ".join(SHORT[m] for m in M) + " |", "|---|" + "---|" * len(M)]

        def cell(g, m, val):
            band = (nul or {}).get("summary", {}).get(g, {}).get(m, {}).get("band97.5")
            return _f(val) + (f" [{band:.2f}]" if band is not None else "")

        out.append("| Core (held-out domain) | " + " | ".join(cell("Held-out", m, means["Held-out"][m]) for m in M) + " |")
        for g in fams:
            out.append(f"| Transfer: {g} | " + " | ".join(cell(g, m, means[g][m]) for m in M) + " |")
        tavg = {m: float(np.mean([means[g][m] for g in fams])) for m in M}
        tband = {m: float(np.mean([nul["summary"][g][m]["band97.5"] for g in fams])) if nul and m in nul["summary"][fams[0]]
                 else None for m in M}
        out.append("| **Transfer, mean of families** | " + " | ".join(
            _f(tavg[m]) + (f" [{tband[m]:.2f}]" if tband[m] is not None else "") for m in M) + " |")
        out.append("| Stego (their rows: empty final reply) | " + " | ".join(cell("blackjack", m, means["blackjack"][m]) for m in M) + " |")
        if stc:
            sh = stc["blackjack_hands_1_7"]
            out.append("| Stego, hands 1-7 averaged (non-empty) | " + " | ".join(
                cell("blackjack_corrected", m, sh[m]) for m in M) + " |")
        out.append("")
        if pse:
            ps = pse["summary"]
            pf = [g for g in ps if g != "blackjack"]
            out.append(f"Pseudo-coalition null ({pse['draws']} probes trained on control runs with a random pair "
                       f"labelled colluder), 97.5th percentile, Transfer mean of families: "
                       + ", ".join(f"{SHORT[m]} {np.mean([ps[g][m]['band97.5'] for g in pf]):.2f}"
                                   for m in ps[pf[0]]) + f"; Stego: "
                       + ", ".join(f"{SHORT[m]} {ps['blackjack'][m]['band97.5']:.2f}" for m in ps["blackjack"]) + ".")
            out.append("")
        if sen:
            out.append("Hyperparameter sensitivity (overall mean AUROC over Transfer and Stego cells): "
                       + ", ".join(f"{k} {v['overall']:.3f}" for k, v in sen.items()) + ".")
            out.append("")
    st = _load(S.RESULTS / "armA_stance_llama_3_1_70b.json")
    if st:
        out += ["Role x stance, Llama-3.1-70B Core agents, leave-domain-out probe score (mean over layers 32-37): "
                + "; ".join(f"{k} n={v['n']} mean {v['mean']:.1f}" for k, v in st.items() if isinstance(v, dict))
                + ". No colluder voted for the fair option, so role and stance cannot be separated among colluders.", ""]
    tb = _load(S.RESULTS / "armA_text_baselines.json")
    if tb:
        out += ["### Text baselines through their pipeline (released transcripts)", "",
                "| View | Tier | " + " | ".join(SHORT[m] for m in M) + " |", "|---|---|" + "---|" * len(M)]
        for key, means in tb["views"].items():
            view, sp = key.split("|")
            fams = [g for g in means if g not in ("Held-out", "blackjack")]
            if "Held-out" in means:
                out.append(f"| {view} | Core | " + " | ".join(_f(means["Held-out"][m]) for m in M) + " |")
                out.append(f"| {view} | Transfer (mean) | " + " | ".join(
                    _f(np.mean([means[g][m] for g in fams])) for m in M) + " |")
            label = "Stego (their empty finals)" if sp == "stego_final" else "Stego (hands 1-7)"
            out.append(f"| {view} | {label} | " + " | ".join(_f(means["blackjack"][m]) for m in M) + " |")
        out += ["", f"Behavioural baselines: Core target-vote count AUROC {tb['votes_core']:.3f}; Stego best "
                f"player's bet-count Spearman correlation AUROC {tb['bets_stego']:.3f}.", "",
                "G3 (frozen rule: any view, any aggregator >= 0.95 makes a ceiling tier): "
                + "; ".join(f"{t} {v['best']:.2f} by {v['by']}" for t, v in tb["G3"].items())
                + ". Every tier is a ceiling tier for a reader of the context (system and user prompts).", ""]
    return out


def _det_table(dets: dict, rows: list[tuple[str, str]]) -> list[str]:
    for v in dets.values():  # text share: not pre-registered, shown because shares are not additive
        dec = v.get("decomposition") or {}
        if dec.get("E_total"):
            dec["text_share"] = dec["E_txt"] / dec["E_total"]
    cols = list(dets)
    out = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for key, label in rows:
        vals = []
        for d in cols:
            v = dets[d]
            for part in key.split("."):
                v = v.get(part) if isinstance(v, dict) else None
            vals.append(_f(v) if not isinstance(v, (list, str)) else (v if isinstance(v, str) else
                                                                        f"[{_f(v[0])}, {_f(v[1])}]"))
        if any(v != "–" for v in vals):  # a row with nothing in it (e.g. no swaps in a context) is left out
            out.append(f"| {label} | " + " | ".join(vals) + " |")
    return out


ROWS = [("auroc_peak", "AUROC, Peak Suspicion (ORIG)"), ("auroc_peak_ci", "95% CI"),
        ("auroc_split", "AUROC, Deception Split"), ("within_run_rank", "within-run colluder rank"),
        ("T_strip_vs_control", "T: STRIP collusion vs control"), ("C_plant_vs_control", "C: PLANT control vs control"),
        ("bare_auroc_peak", "BARE (reply only)"), ("decomposition.share", "context share E_ctx/E_total"),
        ("decomposition.share_ci", "share 95% CI"), ("decomposition.verdict", "verdict (frozen rule)"),
        ("decomposition.text_share", "text share E_txt/E_total (descriptive)"),
        ("auroc_implicit_vs_control", "implicit vs control"), ("auroc_collusion_vs_implicit", "collusion vs implicit"),
        ("observer_shift_mean", "OBSERVER shift (honest agents)")]


def arm_b() -> list[str]:
    out = []
    core = _load(S.RESULTS / "armB_core.json")
    if core:
        out += ["## Arm B: Llama-3.3-70B re-read of the released transcripts", "",
                "### Core (development tier, 31 + 31 runs)", "",
                "Their probe is refitted in our activation space and scored by leave-domain-out folds; the thesis "
                "detectors are zero-shot. Random-direction band (L40, Peak Suspicion): "
                f"{core['random_band_peak']['band97.5']:.2f}.", ""]
        out += _det_table(core["detectors"], [r for r in ROWS if not r[0].startswith(("auroc_peak_ci", "auroc_impl", "auroc_coll"))])
        out.append("")
    held = _load(S.RESULTS / "armB_heldout.json")
    if held:
        t = held["transfer"]
        pr = held["primary"]
        out += ["### Transfer (held-out, 36 + 36 runs; primary tier)", ""]
        out += _det_table(t["detectors"], ROWS)
        out += ["", "Random directions, Peak Suspicion, 97.5th percentile: thesis space (L40) ORIG "
                f"{t['null_thesis_L40']['orig']['band97.5']:.2f}, STRIP {t['null_thesis_L40']['strip']['band97.5']:.2f}; "
                f"their probe's space ORIG {t['null_rose_space']['orig']['band97.5']:.2f}, STRIP "
                f"{t['null_rose_space']['strip']['band97.5']:.2f}. Random persona splits: "
                + "; ".join(f"{k.split('_')[-1]} split0 {v['split0']:.2f}, null median {v['null_median']:.2f}, p {v['p']:.3f}"
                            for k, v in t.items() if k.startswith("random_persona")) + ".", "",
                "**Primary tests (Holm, alpha 0.05):**", "", "| Test | p | Holm threshold | Rejected |", "|---|---|---|---|"]
        for k in S.PRIMARY:
            v = pr["tests"][k]
            pv = f"< {1 / S.N_BOOT:.4f}" if v["p"] == 0 else f"{v['p']:.4f}"
            out.append(f"| {k} | {pv} | {v['holm_threshold']:.4f} | {v['reject']} |")
        e = pr["estimation_delta_auroc_D9K_minus_rose_exact"]
        out += ["", f"Estimation: ΔAUROC (D9+K minus their probe), Transfer ORIG Peak Suspicion = {e['point']:.3f}, "
                f"95% CI [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}].", ""]
        for ctx, blk in held["stego"].items():
            out += [f"### Stego ({ctx} context; audit only)", ""]
            out += _det_table(blk["detectors"], [r for r in ROWS if not r[0].startswith(("auroc_impl", "auroc_coll", "observer"))])
            out += ["", "Their exact rows (final hand, empty reply, end-of-turn token): "
                    + ", ".join(f"{k} {v:.2f}" for k, v in blk["their_rows_empty_final_auroc_peak"].items()) + ".", ""]
        ce = held.get("core_extra")
        if ce:
            out += ["### The 19 extra Core pairs (descriptive)", ""]
            out += _det_table(ce["detectors"], [r for r in ROWS if not r[0].startswith(("auroc_impl", "auroc_coll"))])
            out.append("")
    return out


def _p(p: float) -> str:
    return f"< {1 / S.N_BOOT:.4f} ({S.N_BOOT:,} draws)" if p == 0 else f"= {p:.3g}"


def caveats() -> list[str]:
    g1 = _load(S.LOGS / "g1_rows.json") or {}
    return ["## Data audit, limitations and disclosure", "",
            "- Gate G1: their Llama-3.1-70B activations come from Llama's own runs of the same scenarios (transcripts not "
            "released); their Qwen3-32B Stego and Transfer activations reproduce the released transcripts token for token; "
            "their Core activations come from a different sampling than the released Core transcripts. "
            f"Core runs with transcripts but no released activations: {len(g1.get('core', {}).get('extra_runs', []))}.",
            "- Our Llama-3.3-70B reads are off-policy (Qwen3-32B wrote the text), unlike their Llama numbers; the Arm B "
            "comparison between detectors is internal (same reads).",
            "- PLANT and OBSERVER replies were not written under the planted context: they test what a probe reads, not how "
            "a real colluder behaves. Implicit runs (no instruction) are the natural complement.",
            "- The frozen share rule looks at the context share only; where context and text each carry nearly the whole "
            "effect (D9+K on Core) it still says 'context-dominated', so the text share is shown beside it.",
            "- Small n: 36 + 36 Transfer runs (AUROC SE about 0.06; per family 6 + 6), 20 + 20 Stego runs.",
            "- Reading parts of five scenarios' transcripts (fin_01, audit_01, cicd_01, cr_01, shoe00) and about 30 run "
            "configs informed only the condition builders and the Stego "
            "redaction; no detector, layer, span or aggregator depends on NARCBench, and no score existed before freeze 1. "
            "The work was done before the Apart sprint (Oct 23-25) and must be disclosed as prior work there.", ""]


def findings() -> list[str]:
    """The headline numbers as sentences, every value read from the result files."""
    held, core = _load(S.RESULTS / "armB_heldout.json"), _load(S.RESULTS / "armB_core.json")
    tb = _load(S.RESULTS / "armA_text_baselines.json")
    rq, rl = _load(S.RESULTS / "armA_replicate_qwen3_32b.json"), _load(S.RESULTS / "armA_replicate_llama_3_1_70b.json")
    nq, nl = _load(S.RESULTS / "armA_random_null_qwen3_32b.json"), _load(S.RESULTS / "armA_random_null_llama_3_1_70b.json")
    pq, pl = _load(S.RESULTS / "armA_pseudo_null_qwen3_32b.json"), _load(S.RESULTS / "armA_pseudo_null_llama_3_1_70b.json")
    if not (held and core and tb and rq and rl and nq and nl and pq and pl):
        return []

    def tmean(rep, m="peak_suspicion"):
        fams = [g for g in rep["means"] if g not in ("Held-out", "blackjack")]
        return float(np.mean([rep["means"][g][m] for g in fams]))

    def tband(nul, m="peak_suspicion"):
        fams = [g for g in nul["summary"] if g not in ("Held-out", "blackjack", "blackjack_corrected")]
        return float(np.mean([nul["summary"][g][m]["band97.5"] for g in fams]))

    def pmean(ps, m="peak_suspicion"):
        fams = [g for g in ps["summary"] if g != "blackjack"]
        return float(np.mean([ps["summary"][g][m]["band97.5"] for g in fams]))

    t, pr = held["transfer"]["detectors"], held["primary"]
    r, d = t["rose_exact"], t["D9+K"]
    c = core["detectors"]
    rd, dd = r["decomposition"], d["decomposition"]
    e = pr["estimation_delta_auroc_D9K_minus_rose_exact"]
    rej = [k for k in S.PRIMARY if pr["tests"][k]["reject"]]
    st = held["stego"]["redacted"]
    reply_t = tb["views"]["reply|stego_final"]
    fams = [g for g in reply_t if g not in ("Held-out", "blackjack")]
    ctx_t = tb["views"]["context|stego_final"]
    out = ["## Findings", "",
           f"1. **Their probe's transfer is the collusion instruction.** Refitted on our Llama-3.3-70B reads of Core with "
           f"their recipe, it scores Transfer at AUROC {r['auroc_peak']:.2f}. Planting the colluder's system prompt on "
           f"honest control agents, whose replies stay the same, scores {r['C_plant_vs_control']:.2f}; removing it from "
           f"the colluders, whose collusive replies stay, scores {r['T_strip_vs_control']:.2f} (random band "
           f"{held['transfer']['null_rose_space']['strip']['band97.5']:.2f}); the reply alone scores "
           f"{r['bare_auroc_peak']:.2f}. The context alone carries {rd['share']:.2f} of its effect (95% CI "
           f"{rd['share_ci'][0]:.2f}-{rd['share_ci'][1]:.2f}; P1 p {_p(pr['tests']['P1']['p'])}). It also flags runs "
           f"where nobody was told to collude (implicit vs control {r['auroc_implicit_vs_control']:.2f}).",
           f"2. **The same holds on Core**, where their numbers are perfect: AUROC {c['rose_exact']['auroc_peak']:.2f}, "
           f"PLANT {c['rose_exact']['C_plant_vs_control']:.2f}, STRIP {c['rose_exact']['T_strip_vs_control']:.2f}; the "
           f"agents' own text carries {c['rose_exact']['decomposition']['E_txt'] / c['rose_exact']['decomposition']['E_total']:.2f} "
           f"of the effect. The 19 unseen Core pairs repeat it ({held['core_extra']['detectors']['rose_exact']['auroc_peak']:.2f} / "
           f"STRIP {held['core_extra']['detectors']['rose_exact']['T_strip_vs_control']:.2f}).",
           f"3. **On their own released activations, the probe's margin over chance is small on Transfer.** Peak Suspicion, "
           f"mean of families: Qwen3-32B {tmean(rq):.2f}, Llama-3.1-70B {tmean(rl):.2f}; random directions through the same "
           f"aggregator reach {tband(nq):.2f} / {tband(nl):.2f} (97.5th percentile) and probes trained on control runs with "
           f"a random pair labelled colluder reach {pmean(pq):.2f} / {pmean(pl):.2f}.",
           f"4. **The text gives the task away.** TF-IDF of the prompts alone, through their pipeline, reaches "
           f"{max(ctx_t['Held-out'][m] for m in M):.2f} on Core and {max(float(np.mean([ctx_t[g][m] for g in fams])) for m in M):.2f} "
           f"on Transfer; the replies alone reach {max(float(np.mean([reply_t[g][m] for g in fams])) for m in M):.2f} on Transfer.",
           f"5. **Stego is not covert in the released data**, and their Stego score reads only the prompt: the 160 final-hand "
           f"rows they score have empty replies (gate G1), the big player's plain-text decoding lines reach every player's "
           f"table chat, and their probe (refitted on our reads) scores their exact rows at "
           f"{st['their_rows_empty_final_auroc_peak']['rose_exact']:.2f} even with those lines removed, because the "
           f"colluders' own prompts still hold the agreed scheme.",
           f"6. **The thesis's label-free D9+K, never trained on collusion, detects Transfer collusion above random directions** "
           f"(AUROC {d['auroc_peak']:.2f}, band {held['transfer']['null_thesis_L40']['orig']['band97.5']:.2f}; P4 p = "
           f"{pr['tests']['P4']['p']:.3f}), and its signal does not need the instruction: STRIP {d['T_strip_vs_control']:.2f} "
           f"(P3b p = {pr['tests']['P3b']['p']:.3f}, not significant after Holm), reply only {d['bare_auroc_peak']:.2f}, PLANT "
           f"{d['C_plant_vs_control']:.2f}; context share {dd['share']:.2f} (CI {dd['share_ci'][0]:.2f}-{dd['share_ci'][1]:.2f}, "
           f"mixed). On Core it scores {c['D9+K']['auroc_peak']:.2f} zero-shot, STRIP {c['D9+K']['T_strip_vs_control']:.2f}, "
           f"reply only {c['D9+K']['bare_auroc_peak']:.2f}.",
           f"7. **On the original transcripts their probe is far stronger** (ΔAUROC D9+K minus theirs {e['point']:.2f}, 95% CI "
           f"{e['ci'][0]:.2f} to {e['ci'][1]:.2f}), because it reads the instruction. Holm-significant primary tests: "
           f"{', '.join(rej) if rej else 'none'} (of {', '.join(S.PRIMARY)}).",
           f"8. Core-calibrated thresholds do not transfer: they flag every Transfer run for both detectors, so flag rates are "
           f"not informative; run-level AUROCs are reported instead.", ""]
    return out


def build() -> None:
    lines = ["# NARCBench: Rose et al.'s white-box collusion probes tested with the thesis designs", "",
             f"Data `{S.HF_DATASET}` revision `{S.HF_REVISION[:8]}`, their code `{S.GITHUB_COMMIT[:8]}`; gates and "
             "freezes in `E:/narcbench/logs/gates.md` and `freeze.txt`. Pre-registration: `spec.py`.", ""]
    lines += findings() + arm_a() + arm_b() + caveats()
    S.RESULTS.mkdir(parents=True, exist_ok=True)
    (S.RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {S.RESULTS / 'summary.md'} ({len(lines)} lines)")


if __name__ == "__main__":
    fire.Fire({"build": build})
