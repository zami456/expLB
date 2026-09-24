"""W6C-briefing: O-030 (sarcasm marker rate), O-025 (the Kumar abstract quote),
O-033 (the format-matched control's "+0.03 / clears nothing / two to four times").

O-030 reads the cached lab file cap/run-sarc-pilot/marker_report.json.
O-025 extracts the abstract of papers/Pressure-Testing ... .pdf and diffs it against the
      briefing's block quote.
O-033 is Level 1 only this session: the cap_ctrl matrices are lab-only and the lab PC is
      unreachable, so the per-layer table is taken from the audit's own earlier
      re-derivation from those raw matrices (audit/claims/F-94.md, batch P2-pooled) and the
      ratios the briefing quotes are computed from it here.
"""
import hashlib
import io
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
p = sys.stdout.write


def load(path):
    with io.open(path, "rb") as f:
        b = f.read()
    return json.loads(b.decode("utf-8")), hashlib.md5(b).hexdigest(), len(b)


p("=" * 78 + "\nO-030  sarcasm pilot: marker rate per variant\n" + "=" * 78 + "\n")
mr, md5, n = load("audit/cache/labpc/cap/run-sarc-pilot/marker_report.json")
p("  file audit/cache/labpc/cap/run-sarc-pilot/marker_report.json  md5 %s  %d B  n=%s\n"
  % (md5, n, mr["n"]))
p("  %-22s %s\n" % ("arm", "marker rate per variant v1..v5      overall  mean_count"))
for arm, d in mr["markers"].items():
    p("  %-22s %-36s %.2f     %.2f\n"
      % (arm, str(d["per_variant"]), d["overall"], d["mean_count"]))
p("\n  briefing: 'the marker rate genuinely falls to 0.00 at variants v4 and v5'\n")
for arm, d in mr["markers"].items():
    v4, v5 = d["per_variant"][3], d["per_variant"][4]
    p("    %-22s v4=%.2f v5=%.2f  -> %s\n"
      % (arm, v4, v5, "both 0.00" if v4 == 0 and v5 == 0 else "NOT both 0.00"))
p("\n  the lexical table the briefing prints (ironic vs sycophant):\n")
for s in mr["lexical_subsets"]:
    if s["pair"] == "ironic vs sycophant":
        p("    %-18s raw %.4f  stripped %.4f  n=%d\n"
          % (s["subset"], s["auroc_raw"], s["auroc_stripped"], s["n"]))
p("  briefing: all v1-v5 0.953 / 0.947 ; v4+v5 0.919 / 0.921\n")

p("\n" + "=" * 78 + "\nO-025  the Kumar quotation\n" + "=" * 78 + "\n")
try:
    from pypdf import PdfReader
    r = PdfReader("papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf")
    raw0 = (r.pages[0].extract_text() or "")
    t = re.sub(r"-\n", "", raw0)
    t = re.sub(r"\s+", " ", t)
    i = t.find("decisively rejected")
    p("  PDF papers/Pressure-Testing Deception Probes in LLMs Scaling, Robustness, and the.pdf\n")
    p("  pages: %d ; author line: %s\n" % (len(r.pages), t[:160]))
    p("\n  paper, abstract, verbatim (whitespace normalised):\n    ...%s...\n"
      % t[max(0, i - 40):i + 300])
    p("\n  briefing quotes:\n"
      '    "the single-direction hypothesis is **decisively rejected** (k=1 captures only\n'
      '     0.61-0.80 AUROC of the signal), with cross-domain transfer failure confirmed as\n'
      '     geometric rather than layer-mismatch-driven"\n')
    body = ""
    for pg in r.pages:
        tt = re.sub(r"\s+", " ", re.sub(r"-\n", "", pg.extract_text() or ""))
        if "0.609" in tt:
            j = tt.find("0.609")
            body = tt[max(0, j - 240):j + 120]
            break
    p("\n  the body figure behind 0.61-0.80:\n    ...%s...\n" % body)
except Exception as e:  # pragma: no cover
    p("  PDF read failed: %r\n" % (e,))

p("\n" + "=" * 78 + "\nO-033  the format-matched control (Level 1 -- cap_ctrl matrices are LAB ONLY)\n"
  + "=" * 78 + "\n")
# per-layer values as re-derived from the raw lab matrices by batch P2-pooled and recorded
# in audit/claims/F-94.md (mean pooling, sycophancy_mt target, paper probe, seed 42).
F94 = [
    # L, content(hold-neut), clears?, presence(six-none), clears?
    (0, 0.0524, True, 0.0599, True), (4, 0.0269, False, 0.0299, True),
    (8, 0.0012, False, 0.0005, False), (12, 0.0171, True, 0.0107, False),
    (16, 0.0126, False, 0.0101, False), (20, 0.0519, False, 0.0570, False),
    (22, 0.0519, False, 0.0486, False), (24, 0.0267, False, 0.0316, False),
    (28, 0.0729, False, 0.1293, True), (32, 0.0560, False, 0.1696, True),
    (36, 0.0290, True, 0.1237, True), (40, 0.0247, False, 0.1201, True),
    (44, 0.0277, False, 0.0947, True), (48, 0.0235, False, 0.0755, True),
    (52, 0.0280, False, 0.0829, True), (56, 0.0391, False, 0.0823, True),
    (60, 0.0506, False, 0.0857, True), (64, 0.0395, False, 0.0638, True),
    (68, 0.0278, False, 0.0529, True), (72, 0.0362, False, 0.1059, True),
    (76, 0.0521, False, 0.0993, True),
]
c = [r[1] for r in F94]
pr = [r[3] for r in F94]
p("  source of these 21 rows: audit/claims/F-94.md, script audit/scripts/claims/p2p_F94_lab.py\n"
  "  (the audit's own re-derivation from run-crossgen/kd11/banks/{cap_s23,cap_ctrl}/matrices)\n\n")
p("   L   content  clears   presence  clears   ratio\n")
for L, cc, cs, pp_, ps in F94:
    p("  %3d  %+.4f  %-6s  %+.4f  %-6s  %5.2fx\n"
      % (L, cc, "YES" if cs else "-", pp_, "YES" if ps else "-",
         (pp_ / cc) if cc > 0 else float("nan")))
p("\n  content mean  %+.4f   clears at %d of 21 layers: %s\n"
  % (sum(c) / len(c), sum(1 for r in F94 if r[2]), [r[0] for r in F94 if r[2]]))
p("  presence mean %+.4f   clears at %d of 21 layers\n"
  % (sum(pr) / len(pr), sum(1 for r in F94 if r[4])))
p("  presence/content, mean of means = %.2fx ; per-layer ratios range %.2fx to %.2fx\n"
  % ((sum(pr) / len(pr)) / (sum(c) / len(c)),
     min(r[3] / r[1] for r in F94 if r[1] > 0),
     max(r[3] / r[1] for r in F94 if r[1] > 0)))
p("  briefing: 'Swapping the content costs only ~+0.03 and clears no significance test at any\n"
  "  layer; removing the six entirely costs two to four times that.'\n")
