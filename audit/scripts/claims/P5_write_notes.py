"""Generates audit/claims/D-0xx.md for batch P5-draft-headlines from the batch CSV
plus per-claim excerpts of the script output. Re-runnable; writes only these notes."""
import csv
import os

ROOT = r"C:/Users/SURAIYA/Documents/STUDY/thesis/expLB"
CSVP = f"{ROOT}/audit/phase2/claims_batch_P5-draft-headlines.csv"
OUTD = f"{ROOT}/audit/claims"
FULL = "audit/claims/P5-draft-headlines_script_output.txt"

VERBATIM = {
 "D-003": "On ... HP-KR, which the original study did not evaluate, the published persona contrast inverts, scoring 0.148 AUROC at layer 20.",
 "D-004": "HP-KR's lies are denials of knowledge, and none of the published deceptive personas lies that way.",
 "D-006": "with no benchmark labels, a single contrast direction reaches 0.919 AUROC and 0.910 balanced accuracy at a 1% false-positive rate on HP-KR at layer 40.",
 "D-008": "Across the full train-test matrix of six datasets at layer 40, the extended persona contrast reaches 0.916 mean off-diagonal AUROC, against 0.732 for 8192-dimensional probes each trained on one dataset",
 "D-009": "and beats their average on every held-out dataset",
 "D-010": "the persona principal components add nothing beyond it.",
 "D-027": "It beats their average over training sets on every held-out dataset, at every layer we tested.",
 "D-028": "The 1-d contrast outscores probes on the three leading persona principal components at every depth, by 0.220 AUROC at layer 20 and 0.023 at layer 44.",
 "D-083": "With no benchmark labels, the knowledge-denial contrast reaches 0.890, 0.919 and 0.908 AUROC at layers 20, 40 and 44.",
 "D-093": "At layer 40 it reaches 0.910 balanced accuracy at 1% FPR on the Alpaca control. Its recall is 0.885, and it flags 6% of the honest answers.",
 "D-098": "The extended persona contrast was shaped by no evaluation dataset, and it reaches 0.904, 0.916 and 0.903 off the diagonal at layers 20, 40 and 44, against 0.648, 0.732 and 0.716 for the raw probes.",
 "D-099": "Averaged over training sets, the extended contrast beats the raw probes on every held-out dataset at every layer, and matches or beats the published contrast (Table 7, Appendix B).",
 "D-104": "At 70B, the contrast between group means beats them [persona PCs used as probe features] at every depth (Table 3).",
 "D-117": "At layer 40, averaged over the five Liars' Bench datasets, the extended contrast reaches 0.694 balanced accuracy, 0.413 recall and 0.899 AUROC. The base probe reaches 0.626, 0.281 and 0.778.",
 "D-166": "Persona coordinates scale to a 70B model, but only as far as the persona bank covers the kind of lie being told.",
 "D-167": "Adding roles that deny knowledge turns it into one that, with no benchmark labels, detects those lies at the benchmark's own threshold nearly as well as a probe trained on the benchmark itself.",
 "D-168": "Across datasets, one zero-shot contrast over the extended bank transfers better on average than probes trained on single datasets, and better than the persona subspace.",
 "D-195": "The extended contrast is higher on every test set at every layer.",
}

# section headers of P5_draft_headlines.py output that each claim should quote
EXCERPT = {
 "D-003": ["D-003 / D-083 / D-006 : HP-KR zero-shot AUROC", "A. zero_shot.csv"],
 "D-004": [],
 "D-006": ["D-093 / D-006: knowledge-denial axis, HP-KR, L40, control threshold"],
 "D-008": ["D-008 (abstract) / D-098 (5.2): off-diagonal means", "Off-diagonal mean recomputed from the matrices"],
 "D-009": ["C. Table 7 recomputed from matrix_*.csv"],
 "D-010": ["D-028 / D-104 / D-010 : 1-d contrast vs persona PCs", "Table 3 -- every cell"],
 "D-027": ["C. Table 7 recomputed from matrix_*.csv", "D-009 / D-027 / D-195 verdict inputs"],
 "D-028": ["D-028 / D-104 / D-010 : 1-d contrast vs persona PCs"],
 "D-083": ["D-003 / D-083 / D-006 : HP-KR zero-shot AUROC", "Table 6 (all six datasets x three contrasts)"],
 "D-093": ["D-093 / D-006: knowledge-denial axis, HP-KR, L40, control threshold", "D. zero_shot_operating.csv"],
 "D-098": ["D-008 (abstract) / D-098 (5.2): off-diagonal means"],
 "D-099": ["D-099 second half", "D-009 / D-027 / D-195 verdict inputs"],
 "D-104": ["D-028 / D-104 / D-010 : 1-d contrast vs persona PCs", "B. summary.csv off_diagonal_mean"],
 "D-117": ["D-117: Table 4, Average (Liars' Bench) at the control threshold",
           "Base probe per-dataset at L40 vs L44"],
 "D-166": ["B. summary.csv off_diagonal_mean"],
 "D-167": ["Table 2 balanced-accuracy column", "D. zero_shot_operating.csv"],
 "D-168": ["B. summary.csv off_diagonal_mean"],
 "D-195": ["C. Table 7 recomputed from matrix_*.csv"],
}

EXTRA = {
 "D-004": """Primary sources read during this audit (pypdf text extraction):

papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf, p.9,
"Table 2: Deceptive personas used for contrast construction":

    handler              controls timing, emphasis, and selective disclosure
    broker               uses asymmetric information as leverage
    impression manager   omits damaging context to preserve surface impression
    coverup operator     buries or excludes material facts
    gatekeeper           decides what information passes through
    spin doctor          foregrounds favorable facts and leaves damaging context unspoken
    plausible denier     gives defensible half-disclosures
    narrative manager    shapes the story through sequencing, foregrounding, and omission

  -> 8 deceptive personas; the draft's three quoted phrases are all present verbatim
     (the draft writes "bury or exclude material facts" for "buries or excludes material
     facts"). None of the eight descriptions mentions claiming or feigning not to know.
     Table 3 (7 honest) + Table 4 (5 sycophantic) + Table 5 (6 non-sycophantic, including
     `default`) makes 26 personas, matching the draft's "26 personas of Mahadik and Skapars".

papers/2511.16035v2 (1).pdf (Liars' Bench), p.5:

    "The (2) Harm-Pressure Knowledge Report dataset (HP-KR) contains responses to harmful
     knowledge-report prompts for which a belief was established (labeled deceptive if the
     model denies knowing the answer despite that established belief, and honest otherwise).
     It also contains responses denying knowledge of the answer where no belief could be
     established (labeled as ...)"

  -> "Every lie in HP-KR is therefore a denial of knowledge" is correct by the benchmark's
     own labelling rule. The converse is NOT true: the honest class also contains denials
     (those with no established belief). The draft concedes this in section 6.
""",
 "D-167": """Liars' Bench values quoted by the draft's Table 2 and abstract, checked in
papers/2511.16035v2 (1).pdf during this audit:

  Table 6 (p.54), Mean Probe, row "Llama 3.3 70B", column HP-KR:
      AUROC 0.12   BALANCED ACCURACY AT 1% FPR 0.50      -> draft "0.12" and "0.50"  MATCH
  Table 7 (p.55), Follow-Up Probe, row "Llama 3.3 70B", column HP-KR:
      AUROC 0.14   BALANCED ACCURACY AT 1% FPR 0.26      -> draft "0.14" and "0.26"  MATCH
  Table 8 (p.56), Upper Bound Probe, row "Llama 3.3 70B", column HP-KR:
      BALANCED ACCURACY 0.95   AUROC 0.97                -> draft "0.97" and "0.95"  MATCH

All six externally quoted numbers are correct.
""",
 "D-003": """Cross-check on the ORIGINAL (pre-re-extraction) bank, from his
`git show origin/feat/persona-pc-ood:src/probes/deception-detection/results/persona_ood/zero_shot.csv`:

    harm_refusal_knowledge_report__plain,2139,139,0.10285611510791366,...
                                                  ^ auroc_axis = 0.1029 -> 0.103

The draft (sec 5.1) says "The published grouping inverts on the original bank (0.103 at
layer 20) and still inverts on the re-extracted one (0.148)". BOTH match his committed files.

  RESOLVES Phase-0 red flag RF-3(b): Phase 0 saw "0.103 (draft) vs 0.109 (his
  persona_ood_ext_L20/summary.md:116)". The draft is the one that matches the data; the
  0.109 in his summary.md prose matches no original-bank file. 0.109 is in fact the
  BASE PROBE's HP-KR AUROC at layer 20 (zero_shot_operating.csv, arm "base probe
  (RepE-trained)", control threshold) -- i.e. his summary.md sentence carries the wrong
  number, and the draft does not repeat the error.
""",
 "D-006": """Note on which direction the abstract means.

The abstract's sentence "a single contrast direction reaches 0.919 AUROC and 0.910
balanced accuracy" is the 4-vs-6 KNOWLEDGE-DENIAL contrast. Two sentences later the same
abstract's "the extended persona contrast reaches 0.916 mean off-diagonal AUROC" is the
12-vs-13 EXTENDED contrast. In his own files the extended contrast's balanced accuracy on
HP-KR is 0.500 at every layer (it flags nothing at the Alpaca threshold). The abstract
never names which of the two directions carries the 0.910; only Table 2 disambiguates.
""",
 "D-083": """DID THE LABELS MOVE BETWEEN DRAFT VERSIONS? (asked by the audit lead) -- YES.

His earlier abstract (audit/phase0/collab_abstract_2026-09-21.md, pasted into a session on
2026-09-21) says:

    "Adding four feigned-ignorance personas against six candour personas raises zero-shot
     AUROC to 0.882 and 0.912"

i.e. it attaches 0.882 / 0.912 to the FOUR-vs-SIX contrast. In his committed data:

    persona_ood_ext_L20/zero_shot.csv  HP-KR  auroc_denial_axis = 0.8899   auroc_ext_axis = 0.8816
    persona_ood_ext_L40/zero_shot.csv  HP-KR  auroc_denial_axis = 0.9185   auroc_ext_axis = 0.9124

So 0.882 / 0.912 are the EXTENDED 12-vs-13 numbers, and 0.890 / 0.919 are the 4-vs-6 ones.
The earlier abstract put the extended contrast's numbers under the 4v6 description; the
current draft (Table 2 and sec 5.1) labels both rows correctly and is the version that
matches the committed data. Nothing was re-run: the same two files support both texts.

Other numbers in that earlier abstract, checked against the same files:
    0.148 (8v7 HP-KR L20)                     -> 0.1483   MATCH
    0.593 (8v7 HP-KR L40)                     -> 0.5926   MATCH
    0.043 (sycophancy axis HP-KR L20)         -> 0.042928 MATCH
    0.904 / 0.916 off-diagonal                -> 0.9041 / 0.9158  MATCH
    0.648 / 0.732 raw off-diagonal            -> 0.6478 / 0.7320  MATCH
    0.684 / 0.866 top-3 subspace              -> 0.6837 / 0.8655  MATCH
    "trained probes lose 0.31 and 0.25"       -> 0.3087 / 0.2512  MATCH
    "at layer 40 its first component scores 0.09 below the contrast it is derived from"
        -> ext_pc1 0.7680 vs ext_axis1 0.9158 is 0.148, NOT 0.09.
           The only 0.09-sized gap at L40 is ext_pc3 0.8655 - ext_pc1 0.7680 = 0.0975.
           UNRESOLVED in the earlier abstract; the sentence does not appear in the draft.
""",
 "D-008": """Decomposition of the headline margin (computed from his summary.csv during this audit):

  layer 40:  raw probes            0.7320
             published 8v7 axis    0.8319   (+0.0999 over raw)
             extended 12v13 axis   0.9158   (+0.1838 over raw)
             -> 46% of the margin over raw comes from the 10 added knowledge-denial roles.

  layer 20:  raw 0.6478  ->  8v7 0.6511 (+0.0033)  ->  ext 0.9041 (+0.2564)
             -> ~99% of the margin comes from the added roles.

The added roles are the owner's: his 36-vector bank is byte-identical (md5, all 36) to
run-kd/vectors_s3 on the lab PC. The draft's acknowledgements assign the persona bank to
"[Collaborator name]" (placeholder; the draft carries no author names), so who is credited
cannot be read off the PDF.
""",
}


def load_full():
    with open(f"{ROOT}/{FULL}", encoding="utf-8") as f:
        return f.read().split("\n")


def excerpt(lines, header):
    """Return the block starting at the '=' banner whose title line == header."""
    for i, ln in enumerate(lines):
        if i and lines[i - 1].startswith("======") and ln.strip().startswith(header.strip()):
            out = [lines[i - 1], ln, lines[i + 1]]
            j = i + 2
            while j < len(lines) and not lines[j].startswith("======"):
                out.append(lines[j])
                j += 1
            while out and out[-1].strip() == "":
                out.pop()
            return "\n".join(out)
    return f"[excerpt '{header}' not found]"


def main():
    lines = load_full()
    rows = {r["id"]: r for r in csv.DictReader(open(CSVP, encoding="utf-8"))}
    os.makedirs(OUTD, exist_ok=True)
    for cid, r in rows.items():
        parts = [f"# {cid} — {r['claim_plain'][:90]}",
                 "",
                 "_Batch **P5-draft-headlines** (the collaborator's draft headline claims). "
                 "Checked 2026-09-23._",
                 "",
                 "## The claim",
                 "",
                 f"**Verbatim (draft):** \u201c{VERBATIM.get(cid, '')}\u201d",
                 "",
                 f"**Source:** {r['source_ref']}",
                 "",
                 f"**Restated with its specifics:** {r['claim_plain']}",
                 "",
                 f"**Numbers stated:** {r['stated_numbers']}",
                 "",
                 "## The checking rule for this claim",
                 "",
                 "The collaborator's activations are gone (owner's decision, `audit/PHASE2_PLAN.md`),",
                 "so his pipeline cannot be recomputed. **Level 1** asks only whether the number in the",
                 "draft equals the number in the data he committed on `origin/feat/persona-pc-ood`.",
                 "Where the draft quotes the owner's data or a published paper, that side is checked",
                 "against the raw file or the PDF. A Level-1 result is recorded as",
                 "\u201cmatches his data\u201d, never as \u201choldsx\u201d.".replace("holdsx", "holds"),
                 "",
                 "## What I did",
                 "",
                 f"- Read the evidence file(s) with `git show` (branch never checked out): `{r['evidence_file']}`",
                 f"- Key / row: {r['evidence_key']}",
                 f"- Script: `{r['script']}` (full output: `{FULL}`)",
                 f"- Second level: {r['level2_method']}",
                 "",
                 "## Result",
                 "",
                 f"- **Level 1 value in his file:** {r['level1_value']}",
                 f"- **Level 1 match:** {r['level1_match']}",
                 f"- **Level 2:** {r['level2_value']}",
                 f"- **Version used:** {r['version_used']}",
                 f"- **Robustness:** {r['robustness']}",
                 ""]
        if cid in EXTRA:
            parts += ["## Supporting evidence", "", "```", EXTRA[cid].rstrip(), "```", ""]
        if EXCERPT.get(cid):
            parts += ["## Script output (verbatim excerpt)", ""]
            for h in EXCERPT[cid]:
                parts += ["```", excerpt(lines, h), "```", ""]
        parts += ["## Verdict",
                  "",
                  f"**{r['verdict']}**",
                  "",
                  f"- Confidence: {r['confidence']}",
                  f"- Importance for the paper: {r['importance']}",
                  ""]
        if r["notes"]:
            parts += ["## Notes", "", r["notes"], ""]
        with open(f"{OUTD}/{cid}.md", "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        print("wrote", f"{OUTD}/{cid}.md")


main()
