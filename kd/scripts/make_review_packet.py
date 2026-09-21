#!/usr/bin/env python
"""Generate a self-contained review packet for the kd persona bank.

The bank has to be reviewed by people (or models) with no access to this repo, so the
packet carries its own context: what the personas are for, how they get used, what is
fixed by design, and what specifically needs judging. It is generated from the real
persona and question files so it cannot drift from what will actually run.

    python kd/scripts/make_review_packet.py            # writes kd/docs/REVIEW_PACKET.md
    python kd/scripts/make_review_packet.py --stdout    # print instead
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KD_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = KD_ROOT / "questions" / "kd_questions_shared.jsonl"
ROLE_GROUPS = KD_ROOT / "personas" / "kd_role_groups.json"
INSTRUCTIONS = KD_ROOT / "personas" / "kd_instructions"
OUT = KD_ROOT / "docs" / "REVIEW_PACKET.md"

NEW_GROUPS = ("knowledge_denial", "sincere_ignorance", "plain_assertion")

GROUP_INTENT = {
    "knowledge_denial":
        "The deceptive side. These personas hold knowledge and deny having it. This is the "
        "behaviour the existing persona bank completely lacks.",
    "sincere_ignorance":
        "Honest control A. These say \"I don't know\" and mean it. Because they use the SAME "
        "WORDS as the deceptive group, the axis built from this contrast cannot be a "
        "vocabulary artifact — only sincerity differs. They are written as dispositions — "
        "people who reach for \"I don't know\" by temperament — rather than as accuracy "
        "policies, so they produce that language on any question, not only where a question "
        "hands them a gap.",
    "plain_assertion":
        "Honest control B. These assert knowledge flatly, without hedging. They exist because "
        "the project's existing honest personas are all careful and hedged, so pairing against "
        "them would re-encode \"hedged vs assertive\" instead of \"honest vs deceptive\".",
}

HEADER = """# Review packet — new deception personas

**What this is.** I am building a set of AI "personas" — short system prompts that make a
language model role-play a particular character — for a thesis experiment on detecting
deception inside a model's internal activations. I need a second opinion on whether the
**ten new personas** and **ten new questions** below are well written for their purpose.

You do not need any background in machine learning to help. The core judgement is a
**writing and psychology** one: do these prompts describe the behaviour they claim to, and
would a capable model actually behave that way when given them?

---

## Background: why these personas exist

The experiment tries to detect when a model is lying by reading its internal state rather
than its words. The method works like this:

1. Take a set of personas — some deceptive, some honest.
2. Have the model answer the same questions once per persona, in character.
3. Record the model's internal activations while it answers.
4. Average them per persona, then subtract: `deceptive average − honest average`. That
   difference is a direction in the model's "mind" that supposedly points at deception.
5. Use that direction to score *other* text and ask: does this look like a lie?

This works on most of my test datasets. It fails badly on exactly one, where the model is
asked **"do you know the answer to this question?"** and lies by saying **"I do not know"**
when it does know.

**Why it fails.** My existing deceptive personas are all *narrative managers* — a spin
doctor, a handler, a broker, a PR fixer. They lie by **asserting** things confidently. My
existing honest personas are all *careful record-keepers* — an archivist, a fact-checker,
an ombudsman — who hedge and qualify. So the "deception direction" I built actually learned:

> confident assertion = lying, cautious disclaimer = honest

On the failing dataset the truth is the exact opposite: the *lie* is the cautious-sounding
"I don't know", and the *honest* answer is the confident "yes, I know". So the detector
reads it backwards.

**The fix being tested.** Add personas whose deception *is* feigned ignorance — characters
who claim not to know, not to remember, not to be qualified, when they are. None of my 25
existing personas does this.

---

## Why there are two honest groups (this is the part most worth your scrutiny)

If I pair the new "feigned ignorance" personas against honest personas who speak
**assertively**, the resulting direction might just re-learn "denial words vs assertion
words" — a vocabulary difference, not a deception one. I would have rebuilt the same bug.

So there are two honest control groups:

- **sincere_ignorance** — also says "I don't know", but truthfully. Both sides of this
  contrast use the *same vocabulary*, so any signal must come from sincerity, not wording.
- **plain_assertion** — says "yes, I know that", flatly and unhedged.

Three of the sincere roles deliberately mirror three deceptive ones, so the *role* is held
roughly constant and only the honesty changes:

| deceptive | mirrored honest counterpart |
|---|---|
| `feigned_novice` (pretends not to understand) | `honest_novice` (genuinely at the limit of understanding) |
| `amnesiac_witness` (pretends not to remember) | `scrupulous_witness` (reports memory accurately) |
| `stonewaller` (pretends not to have the info) | `calibrated_reporter` (reports availability accurately) |

---

## How these prompts get used, mechanically

- Each persona has **5 phrasings** of the same character. Each is used as a **system
  prompt** for Llama-3.3-70B-Instruct.
- Each is paired with each of **30 questions** (shown below), which arrive as an ordinary
  user message. The model answers in character.
- That is 5 × 30 = **150 responses per persona**, 5,400 in total.
- A separate model then **grades each response 0–3** on whether it actually stayed in
  character. Only responses scoring 3 ("fully assumes the role, never mentions being an AI")
  are kept. **A persona whose prompts are too weak to hold character produces no usable
  data.**

Style constraints, inherited from the published method this replicates: second person,
roughly 105 characters, five variants, no examples, no formatting instructions. Please keep
suggestions inside that shape.
"""

REVIEW_ASKS = """---

# What I need you to judge

Please go persona by persona and question by question. Concrete rewrites are far more
useful than general approval.

### 1. Will the deceptive personas actually hold character?
This is the biggest practical risk. A model told to "play dumb" often breaks and answers
helpfully anyway, or refuses on the grounds that pretending not to know is dishonest. Which
of the 20 `knowledge_denial` variants look most likely to trigger a refusal or a
break? Would any read as asking the model to deceive a real user?

### 2. Are the two honest groups genuinely distinct?
`sincere_ignorance` and `plain_assertion` must not blur together. Read them back to back:
does `sincere_ignorance` clearly convey *"I say I don't know, and it's true"* rather than
merely *"I'm cautious"*? Does `plain_assertion` read as flat and unhedged rather than
merely confident or arrogant?

### 3. Is `sincere_ignorance` the right mirror for `knowledge_denial`?
The whole design rests on these two groups differing **only** in whether the ignorance is
real. If you can spot some other systematic difference — tone, formality, length,
defensiveness, social register — that is exactly the confound I am trying to avoid, and
finding it now is worth a great deal.

### 4. Do the first six questions force an admit-or-deny choice?
Q20–Q25 put the speaker in a position where they know something and admitting it is costly.
Can any be answered by dodging the knowledge question entirely? Are any two too similar?

### 5. Do the four question types pull their weight?
Six questions put the speaker in a position where they know and admitting is costly. Four
invert it — they genuinely do not know, and admitting *that* is costly. The second set exists
so the sincere personas have honest gaps to admit. Does each of the four actually create that
pressure, or can any be answered without touching the knowledge question?

### 6. Anything that reads as written by someone with a thesis to prove
These were written by me to test a hypothesis I have a stake in. If any wording nudges a
persona toward the behaviour that would make my experiment succeed — rather than describing
the character neutrally — that is a bias I want caught.

---

## What is fixed and should not be changed

- **The first 20 questions.** They are copied verbatim from the published method being
  replicated. Changing them would break comparability.
- **The 26 existing personas.** Same reason.
- **The five-variant, second-person, ~105-character format.**
- **The group structure** (4 deceptive / 3 sincere / 3 assertive) — unless you think a
  specific persona is unsalvageable and should be swapped out.

## Known concerns, so you do not spend time rediscovering them

- The deceptive personas may hold character poorly; I expect a low keep rate and have a
  guard for it, but a prompt that fails more gracefully is better than a guard.
- The three `sincere_ignorance` personas are the newest writing and the least tested. They must
  produce "I don't know" language *by temperament*, not only when a question hands them a gap —
  otherwise they read like the assertive group and the wording-matched contrast is lost.
- The last four questions may tempt a persona into bluffing rather than denying. That is a
  different lie, and I have no persona for it yet.
- `stonewaller` is procedural rather than self-protective, which is intentional — it should
  read as bureaucratic, not evasive-guilty. Tell me if it misses that.
"""


def load_questions() -> list[dict]:
    return [json.loads(l) for l in QUESTIONS.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()

    qs = load_questions()
    groups = json.loads(ROLE_GROUPS.read_text(encoding="utf-8"))["groups"]

    parts = [HEADER]

    parts.append("---\n\n## The 10 new questions\n")
    parts.append(
        "These are the ones to review. They are written in the same first-person voice as the "
        "20 carried-over questions, but where those ask about *disclosing facts*, these ask "
        "about *admitting what you know*. Each describes a moment where the speaker knows "
        "something and saying so is costly.\n")
    for q in qs[20:]:
        parts.append(f"**Q{q['id']}.** {q['question']}\n")

    parts.append(
        "\n<details>\n<summary>The 20 carried-over questions, for context (not under review)"
        "</summary>\n")
    for q in qs[:20]:
        parts.append(f"{q['id']}. {q['question']}")
    parts.append("\n</details>\n")

    parts.append("\n---\n\n## The 10 new personas\n")
    for g in NEW_GROUPS:
        roles = groups[g]["roles"]
        parts.append(f"\n### Group: `{g}` — {len(roles)} personas\n")
        parts.append(f"{GROUP_INTENT[g]}\n")
        for role in roles:
            doc = json.loads((INSTRUCTIONS / f"{role}.json").read_text(encoding="utf-8"))
            desc = doc["_source"]["description_verbatim"]
            parts.append(f"\n**`{role}`** — intended behaviour: *{desc}*\n")
            for i, v in enumerate(doc["instruction"], 1):
                parts.append(f"{i}. {v['pos']}")
            parts.append("")

    parts.append(REVIEW_ASKS)
    text = "\n".join(parts).rstrip() + "\n"

    if args.stdout:
        print(text)
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text, encoding="utf-8")
    words = len(text.split())
    print(f"wrote {OUT}")
    print(f"  {len(text):,} characters, ~{words:,} words")
    print(f"  {len(qs[20:])} questions under review, "
          f"{sum(len(groups[g]['roles']) for g in NEW_GROUPS)} personas, "
          f"{sum(len(groups[g]['roles']) for g in NEW_GROUPS) * 5} instruction variants")


if __name__ == "__main__":
    main()
