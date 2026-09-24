"""Phase 2 step 0: assemble the claim inventory from the SOURCE documents (not from Phase-1 tables).

Sources merged here (mechanical):
  L-*  kd/docs/FINDINGS.md            one row per F-entry (the claim family), with its headline numbers
  D-*  audit/phase0/draft_claims_raw.csv     the collaborator's draft (237 rows)
  A-*  audit/phase0/collab_abstract_2026-09-21.md   his earlier abstract (numbers extracted)
  S-*  audit/phase1/sessions/findings_unrecorded.csv  findings stated in sessions and in no doc

Still to add by hand/agent: his 17 result notes on origin/feat/persona-pc-ood, the owner's other docs
(docs/thesis/*, COLLABORATOR_BRIEFING.md, ROADMAP_TO_DEFENCE.md, kd/docs/REPORT.md) and memory Q1-Q11.

Output: audit/claims_inventory.csv  (id, source, source_ref, claim_short, numbers, layer, dataset_hint, status)
"""
import csv, os, re, subprocess

rows = []


def nums(text, k=8):
    out = re.findall(r"[-+]?\d+\.\d{2,4}|[-+]?0?\.\d{2,4}|\b\d{1,3}\s?%", text)
    seen, keep = set(), []
    for n in out:
        if n not in seen:
            seen.add(n); keep.append(n)
    return "; ".join(keep[:k])


def layer_of(text):
    m = re.findall(r"\bL(\d{1,2})\b|layer[s]?\s+(\d{1,2})", text)
    ls = sorted({int(a or b) for a, b in m})
    return ",".join(str(x) for x in ls[:6])


# --- ledger
md = open("kd/docs/FINDINGS.md", encoding="utf-8", errors="replace").read().splitlines()
cur = None
for i, line in enumerate(md, 1):
    m = re.match(r"^#{1,4}\s+\**\s*(F-\d+)\b(.*)$", line)
    if m:
        if cur:
            rows.append(cur)
        title = re.sub(r"^[·\s*]+", "", m.group(2)).strip(" *")
        cur = dict(id="L-" + m.group(1), source="FINDINGS.md", source_ref=f"kd/docs/FINDINGS.md:{i}",
                   claim_short=title[:200], numbers="", layer="", dataset_hint="", status="TODO", body="")
    elif cur and len(cur["body"]) < 4000:
        cur["body"] += " " + line
if cur:
    rows.append(cur)
for r in rows:
    r["numbers"] = nums(r["claim_short"] + " " + r["body"])
    r["layer"] = layer_of(r["claim_short"] + " " + r["body"][:800])
    for d in ("harm-pressure-knowledge-report", "HP-KR", "sycophancy_mt", "convincing-game", "instructed-deception",
              "insider-trading", "ai_liar", "roleplaying", "mask", "RepE", "alpaca"):
        if d.lower() in (r["claim_short"] + " " + r["body"][:1200]).lower():
            r["dataset_hint"] = (r["dataset_hint"] + " " + d).strip()
    r.pop("body")

# --- draft
if os.path.exists("audit/phase0/draft_claims_raw.csv"):
    for r in csv.DictReader(open("audit/phase0/draft_claims_raw.csv", encoding="utf-8")):
        cid = r.get("claim_id") or r.get("id") or f"D-{len(rows)}"
        text = r.get("claim") or r.get("claim_verbatim") or r.get("text") or ""
        rows.append(dict(id="D-" + cid.replace("D-", ""), source="draft PDF",
                         source_ref=f"draft p{r.get('page', '?')} {r.get('section', '')}".strip(),
                         claim_short=text[:200], numbers=(r.get("numbers") or nums(text)),
                         layer=(r.get("layer") or layer_of(text)), dataset_hint=(r.get("datasets") or r.get("dataset") or ""),
                         status="TODO"))

# --- abstract
if os.path.exists("audit/phase0/collab_abstract_2026-09-21.md"):
    txt = open("audit/phase0/collab_abstract_2026-09-21.md", encoding="utf-8").read()
    body = txt.split("**Abstract:**", 1)[-1]
    for j, sent in enumerate([s.strip() for s in re.split(r"(?<=[.!?])\s+", body) if len(s.strip()) > 60], 1):
        if re.search(r"\d", sent):
            rows.append(dict(id=f"A-{j:02d}", source="abstract 2026-09-21",
                             source_ref="audit/phase0/collab_abstract_2026-09-21.md",
                             claim_short=sent[:200], numbers=nums(sent), layer=layer_of(sent),
                             dataset_hint="", status="TODO"))

# --- session-only findings
pth = "audit/phase1/sessions/findings_unrecorded.csv"
if os.path.exists(pth):
    for j, r in enumerate(csv.DictReader(open(pth, encoding="utf-8")), 1):
        text = r.get("claim") or ""
        rows.append(dict(id=f"S-{j:02d}", source="session (unrecorded)",
                         source_ref=f"{r.get('session', '')}:{r.get('line', '')}",
                         claim_short=text[:200], numbers=(r.get("numbers") or nums(text)),
                         layer=layer_of(text), dataset_hint="", status="TODO"))

with open("audit/claims_inventory.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["id", "source", "source_ref", "claim_short", "numbers", "layer", "dataset_hint", "status"])
    w.writeheader()
    for r in rows:
        w.writerow(r)
import collections
print("claim inventory:", len(rows), "rows")
print(dict(collections.Counter(r["source"] for r in rows)))
