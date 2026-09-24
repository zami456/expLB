"""Build audit/papers.csv from papers/ PDFs, docs/papers notes, and papers_cited_raw.tsv (+ the draft's reference list)."""
import csv, collections
present = {  # arXiv id -> (pdf path, title from its first page — VERIFIED by p1 first-page read)
 "2511.16035": ("papers/2511.16035v2 (1).pdf", "Liars' Bench: Evaluating Lie Detectors for Language Models (Kretschmar et al.)"),
 "2502.03407": ("papers/Detecting Strategic Deception Using Linear.pdf", "Detecting Strategic Deception Using Linear Probes (Goldowsky-Dill et al.)"),
 "2605.09391": ("papers/Do Linear Probes Generalize Better in Persona Coordinates.pdf", "Do Linear Probes Generalize Better in Persona Coordinates? (Mahadik & Skapars)"),
 "2309.15840": ("papers/HOW TO CATCH AN AI LIAR.pdf", "How to Catch an AI Liar (Pacchiardi et al.)"),
 "2605.27958": ("papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf", "Pressure-Testing Deception Probes in LLMs (Kumar)"),
 "2609.02893": ("papers/Probe Generalization as Subspace Selection for OOD Deception Detectio.pdf", "Probe Generalization as Subspace Selection for OOD Deception Detection (Yoo & Skapars)"),
 "2601.10387": ("papers/The Assistant Axis Situating and Stabilizing.pdf", "The Assistant Axis (Lu et al.)"),
 "2511.17408": ("papers/The Impact of Off-Policy Training Data.pdf", "The Impact of Off-Policy Training Data on Probe Generalisation (Kirch et al.)"),
 "2503.03750": ("papers/The MASK Benchmark Disentangling.pdf", "The MASK Benchmark (Ren et al.)"),
 "2606.15821": ("papers/The Truth Stays in the Family.pdf", "The Truth Stays in the Family (Choi et al.)"),
 "2602.20273": ("papers/The Truthfulness Spectrum Hypothesis.pdf", "The Truthfulness Spectrum Hypothesis (Ying et al.)"),
}
notes = {"2511.16035": "docs/papers/liars-bench-notes.md", "2502.03407": "docs/papers/deception-probes-notes.md",
         "2605.09391": "docs/papers/persona-coordinates-notes.md", "2309.15840": "docs/papers/pacchiardi-notes.md",
         "2601.10387": "docs/papers/assistant-axis-notes.md", "2503.03750": "docs/papers/mask-notes.md",
         "2602.20273": "docs/papers/truthfulness-spectrum-notes.md"}
titles = {  # titles of cited-but-missing papers, as written at the citing location (DOC-ONLY)
 "2310.06824": "The Geometry of Truth (Marks & Tegmark)", "1610.01644": "Understanding intermediate layers using linear classifier probes (Alain & Bengio)",
 "2602.01425": "One probe won't catch them all: Towards targeted deception detection (Natarajan et al.)",
 "2401.05566": "Sleeper Agents (Hubinger et al., Anthropic)", "2407.14561": "NNsight and NDIF (Fiotto-Kaufman et al.)",
 "2311.07590": "LLMs can strategically deceive their users when put under pressure (Scheurer et al.)",
 "2411.00640": "Miller — clustered standard errors for evals (cited in NOVELTY_ASSESSMENT; exact title not in context)",
 "2509.21305": "? — separates sycophantic agreement / praise / genuine agreement (title not in context; NOT identified)",
 "2407.21783": "The Llama 3 Herd of Models (Grattafiori et al.)", "2106.09685": "LoRA (Hu et al.)",
 "2310.13548": "Towards Understanding Sycophancy in Language Models (Sharma et al.)",
 "2212.03827": "Discovering latent knowledge in language models without supervision (Burns et al.)",
 "2507.21509": "Persona vectors (Chen et al.)", "2403.03218": "The WMDP benchmark (Li et al.)",
 "2310.01405": "Representation engineering (Zou et al.)"}
draft_noid = ["Azaria & Mitchell 2023, The internal state of an LLM knows when it's lying (EMNLP Findings) — no arXiv id in draft",
              "Birardi et al. 2026, Aletheia's Quest retrospective (EleutherAI blog) — no arXiv id",
              "Taori et al. 2023, Stanford Alpaca (GitHub) — not a paper"]
rows = list(csv.DictReader(open("audit/phase1/papers_cited_raw.tsv", encoding="utf-8"), delimiter="\t"))
cited = {r["arxiv_id"]: r for r in rows}
draft_ids = set()
for line in open("audit/phase0/draft_text.txt", encoding="utf-8"):
    import re
    draft_ids |= set(re.findall(r"arXiv:(\d{4}\.\d{4,5})", line))
with open("audit/papers.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["arxiv_id", "title", "pdf_present", "pdf_path", "notes_present", "notes_path", "cited_in_draft", "n_citations_found", "first_citation_location", "status"])
    for aid in sorted(set(present) | set(cited), key=lambda a: (a not in present, a)):
        pdf, t = present.get(aid, ("", titles.get(aid, "")))
        c = cited.get(aid)
        st = "PDF+NOTES" if aid in present and aid in notes else ("PDF, NO NOTES" if aid in present else "CITED, NO PDF")
        w.writerow([aid, t, "y" if aid in present else "n", pdf, "y" if aid in notes else "n", notes.get(aid, ""),
                    "y" if aid in draft_ids else "n", c["n_citations"] if c else 0, c["first_location"] if c else "", st])
    for d in draft_noid:
        w.writerow(["", d, "n", "", "n", "", "y", "", "draft references", "CITED, NO PDF (no arXiv id)"])
print(open("audit/papers.csv", encoding="utf-8").read())
