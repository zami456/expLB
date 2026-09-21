#!/usr/bin/env python
"""Render the review packet as a standalone HTML page for sharing.

Same content as make_review_packet.py, but laid out so the comparison that matters is
structural: the three mirrored persona pairs sit side by side, because the review's hardest
question is whether a deceptive persona and its honest mirror differ in anything BESIDES
sincerity. Markdown cannot put them next to each other; a page can.

Generated from the real persona and question files, so it cannot drift from what will run.

    python kd/scripts/make_review_page.py        # -> kd/docs/review_page.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

KD_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = KD_ROOT / "questions" / "kd_questions_shared.jsonl"
ROLE_GROUPS = KD_ROOT / "personas" / "kd_role_groups.json"
INSTRUCTIONS = KD_ROOT / "personas" / "kd_instructions"
OUT = KD_ROOT / "docs" / "review_page.html"

# deceptive role -> its honest mirror. The pairing is the design's core claim: hold the role
# roughly constant, vary only whether the ignorance is real.
MIRRORS = [
    ("pc_feigned_novice", "pc_honest_novice",
     "pretends not to understand", "genuinely at the limit of understanding"),
    ("pc_amnesiac_witness", "pc_scrupulous_witness",
     "pretends not to remember", "reports memory accurately"),
    ("pc_stonewaller", "pc_calibrated_reporter",
     "pretends not to have the information", "reports availability accurately"),
]
UNPAIRED_DECEPTIVE = ["pc_deniable_expert"]
ASSERTION = ["pc_forthright_expert", "pc_candid_witness", "pc_plain_dealer"]

E = html.escape


def load():
    qs = [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]
    roles = {}
    for f in INSTRUCTIONS.glob("pc_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        roles[f.stem] = {
            "desc": d["_source"].get("description_verbatim", ""),
            "variants": [v["pos"] for v in d["instruction"]],
            "group": d["_source"].get("group", ""),
        }
    return qs, roles


def persona_block(slug: str, roles: dict, side: str, note: str = "") -> str:
    r = roles[slug]
    lis = "\n".join(f"<li>{E(v)}</li>" for v in r["variants"])
    notehtml = f'<p class="mirror-note">{E(note)}</p>' if note else ""
    return f"""<article class="persona {side}">
  <h4><code>{E(slug)}</code></h4>
  {notehtml}
  <p class="intent"><span class="lbl">Intended behaviour</span>{E(r['desc'])}</p>
  <ol class="variants">
{lis}
  </ol>
</article>"""


def build(qs, roles) -> str:
    new_q = "\n".join(
        f'<li><span class="qn">Q{q["id"]}</span><p>{E(q["question"])}</p></li>' for q in qs[20:])
    old_q = "\n".join(f"<li>{E(q['question'])}</li>" for q in qs[:20])

    pairs = []
    for dec, hon, dnote, hnote in MIRRORS:
        pairs.append(f"""<div class="pair">
{persona_block(dec, roles, "deceptive", dnote)}
{persona_block(hon, roles, "honest", hnote)}
</div>""")
    pairs_html = "\n".join(pairs)
    unpaired = "\n".join(persona_block(s, roles, "deceptive") for s in UNPAIRED_DECEPTIVE)
    assertion = "\n".join(persona_block(s, roles, "honest") for s in ASSERTION)

    return f"""<title>Knowledge-Denial Personas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
  :root {{
    --ground:#F6F7F9; --surface:#FFFFFF; --sunken:#EDEFF3;
    --ink:#181B21; --ink-2:#3C434F; --muted:#666E7E; --rule:#DDE1E8;
    --oxide:#A2432C; --oxide-soft:#F6EAE6; --oxide-rule:#E2C3B9;
    --teal:#22655A; --teal-soft:#E6F0ED; --teal-rule:#B6D3CC;
    --serif:'Spectral',Georgia,'Times New Roman',serif;
    --sans:'IBM Plex Sans',system-ui,-apple-system,'Segoe UI',sans-serif;
    --mono:'IBM Plex Mono',ui-monospace,'SFMono-Regular',Menlo,monospace;
  }}
  @media (prefers-color-scheme:dark) {{
    :root:not([data-theme="light"]) {{
      --ground:#111318; --surface:#181B22; --sunken:#1F232B;
      --ink:#E9EBF0; --ink-2:#C2C8D2; --muted:#949CAB; --rule:#2B313B;
      --oxide:#E29173; --oxide-soft:#2A1C17; --oxide-rule:#4A2E23;
      --teal:#6FC2B0; --teal-soft:#132523; --teal-rule:#26463F;
    }}
  }}
  :root[data-theme="dark"] {{
    --ground:#111318; --surface:#181B22; --sunken:#1F232B;
    --ink:#E9EBF0; --ink-2:#C2C8D2; --muted:#949CAB; --rule:#2B313B;
    --oxide:#E29173; --oxide-soft:#2A1C17; --oxide-rule:#4A2E23;
    --teal:#6FC2B0; --teal-soft:#132523; --teal-rule:#26463F;
  }}
  *{{box-sizing:border-box}}
  body{{
    background:var(--ground); color:var(--ink); font-family:var(--sans);
    font-size:16px; line-height:1.62; margin:0; padding-block:0 4rem;
  }}
  .wrap{{max-width:1080px; margin:0 auto; padding-inline:16px}}
  .col{{max-width:68ch}}
  h1,h2,h3,h4{{font-family:var(--serif); text-wrap:balance; margin:0}}
  h1{{font-size:clamp(1.9rem,5vw,2.9rem); line-height:1.12; font-weight:600; letter-spacing:-.01em}}
  h2{{font-size:clamp(1.35rem,3vw,1.7rem); font-weight:600; margin-top:3.2rem; margin-bottom:.9rem}}
  h3{{font-size:1.12rem; font-weight:600; margin-top:2rem; margin-bottom:.5rem}}
  h4{{font-size:1rem; font-weight:600}}
  p{{margin:0 0 1rem}}
  a{{color:var(--teal)}}
  code{{font-family:var(--mono); font-size:.88em}}
  strong{{font-weight:600}}

  header.masthead{{
    border-bottom:2px solid var(--ink); padding-block:2.6rem 1.2rem; margin-bottom:2.2rem;
  }}
  .eyebrow{{
    font-family:var(--mono); font-size:.72rem; letter-spacing:.14em; text-transform:uppercase;
    color:var(--muted); margin:0 0 .9rem;
  }}
  .standfirst{{font-family:var(--serif); font-size:1.18rem; color:var(--ink-2); margin-top:1rem}}
  .facts{{
    display:flex; flex-wrap:wrap; gap:.5rem 1.6rem; margin-top:1.4rem; padding-top:1.1rem;
    border-top:1px solid var(--rule); font-family:var(--mono); font-size:.78rem; color:var(--muted);
  }}
  .facts b{{color:var(--ink); font-weight:500}}

  .callout{{
    background:var(--sunken); border-left:3px solid var(--oxide);
    padding:1.1rem 1.3rem; margin:1.6rem 0; border-radius:0 4px 4px 0;
  }}
  .callout p:last-child{{margin-bottom:0}}
  .callout .rule-said{{
    font-family:var(--serif); font-style:italic; font-size:1.1rem; color:var(--ink);
  }}

  ol.steps{{counter-reset:s; list-style:none; padding:0; margin:1.2rem 0}}
  ol.steps li{{
    counter-increment:s; position:relative; padding-left:2.4rem; margin-bottom:.7rem;
  }}
  ol.steps li::before{{
    content:counter(s); position:absolute; left:0; top:.1rem;
    font-family:var(--mono); font-size:.75rem; color:var(--muted);
    border:1px solid var(--rule); border-radius:50%; width:1.55rem; height:1.55rem;
    display:grid; place-items:center;
  }}

  table{{border-collapse:collapse; width:100%; font-size:.93rem}}
  .scroll{{overflow-x:auto; margin:1.2rem 0}}
  th,td{{text-align:left; padding:.6rem .8rem; border-bottom:1px solid var(--rule); vertical-align:top}}
  th{{font-family:var(--mono); font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; color:var(--muted); font-weight:500}}

  ol.qs{{list-style:none; padding:0; margin:1.4rem 0; counter-reset:q}}
  ol.qs li{{display:flex; gap:1rem; padding:.85rem 0; border-bottom:1px solid var(--rule)}}
  ol.qs li p{{margin:0; font-family:var(--serif); font-size:1.05rem}}
  .qn{{font-family:var(--mono); font-size:.75rem; color:var(--muted); flex:0 0 2.6rem; padding-top:.32rem}}

  .pair{{display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-bottom:1rem}}
  .persona{{
    background:var(--surface); border:1px solid var(--rule); border-radius:5px;
    padding:1.1rem 1.2rem;
  }}
  .persona.deceptive{{border-top:3px solid var(--oxide)}}
  .persona.honest{{border-top:3px solid var(--teal)}}
  .persona h4 code{{font-size:.95rem}}
  .persona.deceptive h4 code{{color:var(--oxide)}}
  .persona.honest h4 code{{color:var(--teal)}}
  .mirror-note{{font-size:.84rem; color:var(--muted); margin:.3rem 0 0; font-style:italic}}
  .intent{{font-size:.9rem; color:var(--ink-2); margin:.8rem 0 .9rem}}
  .lbl{{
    display:block; font-family:var(--mono); font-size:.66rem; letter-spacing:.1em;
    text-transform:uppercase; color:var(--muted); margin-bottom:.15rem;
  }}
  ol.variants{{margin:0; padding-left:1.25rem; font-size:.92rem}}
  ol.variants li{{margin-bottom:.5rem; color:var(--ink-2)}}
  ol.variants li::marker{{font-family:var(--mono); font-size:.78rem; color:var(--muted)}}

  .solo{{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:1rem}}

  .asks{{margin-top:1.4rem}}
  .ask{{
    border-top:1px solid var(--rule); padding:1.15rem 0;
    display:grid; grid-template-columns:2.6rem 1fr; gap:1rem;
  }}
  .ask .num{{font-family:var(--mono); font-size:1.3rem; color:var(--oxide); line-height:1.2}}
  .ask h3{{margin:0 0 .35rem; font-size:1.05rem}}
  .ask p{{margin:0; color:var(--ink-2); font-size:.96rem}}

  details{{
    background:var(--sunken); border-radius:5px; padding:.85rem 1.1rem; margin:1.4rem 0;
  }}
  summary{{cursor:pointer; font-family:var(--mono); font-size:.8rem; color:var(--muted)}}
  details ol{{margin:.9rem 0 .2rem; padding-left:1.4rem; font-size:.93rem; color:var(--ink-2)}}
  details li{{margin-bottom:.4rem}}

  .fixed-list li{{margin-bottom:.45rem}}
  footer{{margin-top:3.5rem; padding-top:1.2rem; border-top:1px solid var(--rule);
    font-family:var(--mono); font-size:.76rem; color:var(--muted)}}

  @media (max-width:720px){{
    .pair{{grid-template-columns:1fr}}
    .ask{{grid-template-columns:1.9rem 1fr; gap:.7rem}}
  }}
  @media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}
  :focus-visible{{outline:2px solid var(--teal); outline-offset:2px}}
</style>

<div class="wrap">
<header class="masthead">
  <p class="eyebrow">Review request &middot; persona prompts</p>
  <h1>Ten personas that lie by claiming not to know</h1>
  <p class="standfirst col">I need a second opinion on whether these prompts actually describe
  the behaviour they claim to &mdash; and whether a capable model would really behave that way
  when given them.</p>
  <div class="facts">
    <span><b>10</b> new personas</span>
    <span><b>50</b> instruction variants</span>
    <span><b>10</b> new questions</span>
    <span><b>5,400</b> responses once it runs</span>
  </div>
</header>

<div class="col">
<p><strong>You do not need a machine-learning background to help.</strong> The core judgement
is a writing and psychology one. Everything needed to make it is on this page.</p>

<h2>Why these personas exist</h2>
<p>The experiment tries to detect when a language model is lying by reading its internal
state rather than its words. The method:</p>
<ol class="steps">
  <li>Take a set of personas &mdash; some deceptive, some honest.</li>
  <li>Have the model answer the same questions once per persona, in character.</li>
  <li>Record the model's internal activations while it answers.</li>
  <li>Average per persona, then subtract: <em>deceptive average &minus; honest average</em>.
      That difference is a direction that supposedly points at deception.</li>
  <li>Use it to score other text: does this look like a lie?</li>
</ol>
<p>This works on most of my test datasets. It fails badly on exactly one &mdash; where the
model is asked <em>&ldquo;do you know the answer to this question?&rdquo;</em> and lies by
saying <em>&ldquo;I do not know&rdquo;</em> when it does know.</p>

<div class="callout">
  <p class="rule-said">My existing deceptive personas all lie by asserting. My existing honest
  personas all hedge. So the direction I built actually learned: confident assertion = lying,
  cautious disclaimer = honest.</p>
  <p style="margin-top:.8rem">On the failing dataset the truth is exactly inverted &mdash; the
  <em>lie</em> is the cautious-sounding &ldquo;I don't know&rdquo;, and the honest answer is the
  confident &ldquo;yes, I know&rdquo;. So the detector reads it backwards.</p>
</div>

<p>The fix being tested: add personas whose deception <em>is</em> feigned ignorance &mdash;
characters who claim not to know, not to remember, not to be qualified, when they are. None of
my 25 existing personas does this.</p>

<h2>Why there are two honest groups</h2>
<p>This is the part most worth your scrutiny. If I pair the new feigned-ignorance personas
against honest personas who speak <em>assertively</em>, the resulting direction might just
re-learn &ldquo;denial words vs assertion words&rdquo; &mdash; a vocabulary difference, not a
deception one. I would have rebuilt the same bug.</p>
<p>So there are two honest controls. <strong>Sincere ignorance</strong> also says &ldquo;I don't
know&rdquo;, but truthfully &mdash; both sides use the same vocabulary, so any signal must come
from sincerity. <strong>Plain assertion</strong> says &ldquo;yes, I know that&rdquo;, flatly.</p>
<p>Three deceptive roles have a deliberate honest mirror, so the <em>role</em> is held roughly
constant and only the honesty changes. They are shown side by side below.</p>

<h2>How these prompts get used</h2>
<p>Each persona has five phrasings of the same character; each is used as a <strong>system
prompt</strong> for Llama-3.3-70B-Instruct, paired with each of 30 questions arriving as an
ordinary user message. That is 150 responses per persona, 5,400 in total.</p>
<p>A separate model then grades each response 0&ndash;3 on whether it stayed in character. Only
responses scoring 3 &mdash; <em>fully assumes the role, never mentions being an AI</em> &mdash;
are kept. <strong>A persona whose prompts are too weak to hold character produces no usable
data.</strong></p>
<p>Style constraints, inherited from the published method being replicated: second person,
roughly 105 characters, five variants, no examples, no formatting instructions. Please keep
suggestions inside that shape.</p>

<h2>The 10 new questions</h2>
<p>These are written in the same first-person voice as the 20 carried-over questions, but where
those ask about <em>disclosing facts</em>, these ask about <em>admitting what you know</em>. Each
describes a moment where the speaker knows something and saying so is costly.</p>
</div>

<ol class="qs">
{new_q}
</ol>

<div class="col">
<details>
  <summary>The 20 carried-over questions, for context &mdash; not under review</summary>
  <ol>
{old_q}
  </ol>
</details>
</div>

<h2>Mirrored pairs</h2>
<div class="col">
<p>Deceptive on the left, its honest mirror on the right. The design rests on these differing
<strong>only</strong> in whether the ignorance is real. Any other systematic difference &mdash;
tone, length, defensiveness, formality &mdash; is a confound, and finding one now is worth a
great deal.</p>
</div>
{pairs_html}

<h2>Deceptive, unpaired</h2>
<div class="solo">
{unpaired}
</div>

<h2>Plain assertion &mdash; honest control B</h2>
<div class="col">
<p>These exist because the project's existing honest personas are all careful and hedged, so
pairing against them would re-encode &ldquo;hedged vs assertive&rdquo; instead of
&ldquo;honest vs deceptive&rdquo;. They should read flat and unhedged &mdash; not arrogant.</p>
</div>
<div class="solo">
{assertion}
</div>

<div class="col">
<h2>What I need you to judge</h2>
<p>Please go persona by persona and question by question. Concrete rewrites are far more useful
than general approval.</p>
</div>

<div class="asks col">
  <div class="ask"><span class="num">1</span><div>
    <h3>Will the deceptive personas hold character?</h3>
    <p>The biggest practical risk. A model told to play dumb often breaks and answers helpfully
    anyway, or refuses on the grounds that pretending not to know is dishonest. Which of the 20
    knowledge-denial variants look most likely to trigger a refusal or a break? Would any read as
    asking the model to deceive a real user?</p></div></div>
  <div class="ask"><span class="num">2</span><div>
    <h3>Are the two honest groups genuinely distinct?</h3>
    <p>Read them back to back. Does sincere ignorance clearly convey <em>&ldquo;I say I don't
    know, and it's true&rdquo;</em> rather than merely &ldquo;I'm cautious&rdquo;? Does plain
    assertion read as flat and unhedged rather than confident or arrogant?</p></div></div>
  <div class="ask"><span class="num">3</span><div>
    <h3>Is sincere ignorance the right mirror?</h3>
    <p>Each pair must differ only in whether the ignorance is real. If you can spot some other
    systematic difference, that is exactly the confound I am trying to avoid.</p></div></div>
  <div class="ask"><span class="num">4</span><div>
    <h3>Do the 10 questions force an admit-or-deny choice?</h3>
    <p>Each should put the speaker in a position where they know something and must decide
    whether to say so. Can any be answered by dodging the knowledge question entirely? Are any
    two too similar? I already suspect Q23 and Q25 overlap on expertise.</p></div></div>
  <div class="ask"><span class="num">5</span><div>
    <h3>Do the four question types pull their weight?</h3>
    <p>Six questions put the speaker in a position where they know and admitting is costly; four
    invert it &mdash; they genuinely do not know, and admitting <em>that</em> is costly. The second
    set exists so the sincere personas have honest gaps to admit. Does each of the four actually
    create that pressure, or can any be answered without touching the knowledge question?</p>
    </div></div>
  <div class="ask"><span class="num">6</span><div>
    <h3>Anything that reads as written by someone with a thesis to prove</h3>
    <p>I wrote these to test a hypothesis I have a stake in. If any wording nudges a persona
    toward the behaviour that would make my experiment succeed &mdash; rather than describing the
    character neutrally &mdash; that is a bias I want caught.</p></div></div>
</div>

<div class="col">
<h2>What is fixed</h2>
<ul class="fixed-list">
  <li><strong>The first 20 questions.</strong> Copied verbatim from the published method being
      replicated; changing them breaks comparability.</li>
  <li><strong>The 26 existing personas.</strong> Same reason.</li>
  <li><strong>The five-variant, second-person, ~105-character format.</strong></li>
  <li><strong>The group structure</strong> (4 deceptive / 3 sincere / 3 assertive) &mdash; unless
      a persona is unsalvageable and should be swapped out.</li>
</ul>

<h2>Known concerns</h2>
<p>So you don't spend time rediscovering them:</p>
<ul class="fixed-list">
  <li>The deceptive personas may hold character poorly. I expect a low keep rate and have a
      guard, but a prompt that fails more gracefully beats a guard.</li>
  <li>The three <code>sincere_ignorance</code> personas are the newest writing and the least
      tested. They must produce &ldquo;I don't know&rdquo; language <em>by temperament</em>, not
      only when a question hands them a gap &mdash; otherwise they read like the assertive
      group and the wording-matched contrast is lost.</li>
  <li>The last four questions may tempt a persona into bluffing rather than denying. That is a
      different lie, and I have no persona for it yet.</li>
  <li><code>stonewaller</code> is procedural rather than self-protective, which is intentional
      &mdash; it should read bureaucratic, not evasive-guilty. Tell me if it misses that.</li>
</ul>

<footer>
  Generated from the live persona bank &mdash; 36 personas, 30 shared questions.
  Nothing has run yet; this is the review gate before any compute.
</footer>
</div>
</div>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    qs, roles = load()
    missing = [s for s, _, _, _ in MIRRORS] + [h for _, h, _, _ in MIRRORS] \
        + UNPAIRED_DECEPTIVE + ASSERTION
    absent = [s for s in missing if s not in roles]
    if absent:
        raise SystemExit(f"missing persona files: {absent}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(qs, roles), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size:,} bytes, {len(missing)} personas)")


if __name__ == "__main__":
    main()
