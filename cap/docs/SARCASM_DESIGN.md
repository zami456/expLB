# The sarcasm bank — design, for review before anything is generated

**Status: draft. Nothing generated.** Posting this for review first, because the design decision
below is the whole experiment and it is not obvious I have it right.

---

## What it is for

Every test of our register claim so far is **negative**: persona axes *fail* where register and
the labelled construct come apart (0.300 on knowledge-report, 0.374 on capitulation, both beaten
by a random direction). A sarcasm axis would be the first **positive** test — build the confound
itself and show it explains the failure.

Sarcasm is the right confound for a specific reason: **it is saying something you don't mean,
without intending to mislead.** Deception minus the intent. So a sarcasm direction tests whether
our "deception" axes detect *intent* or merely *literal-meaning mismatch*.

Goldowsky-Dill et al. concede their own probe partly fires on *"is this scenario
deception-related"* rather than on deceptive intent. Kirch et al. formalise the distinction as
**text-evident vs text-ambiguous**. A sarcasm axis puts a number on it: if
`cos(d_sarcasm, d_deception)` is high, the deception axis is substantially a mismatch detector.

---

## The problem that kills the naive version

A persona told to "be sarcastic" writes *"oh, brilliant"*, *"sure, that'll work"*, scare quotes,
exclamation marks. The axis then separates those tokens, not sarcasm. We would have built a
lexical detector and called it a representation — the exact failure the knowledge-denial bank was
restructured to avoid, and the exact failure our own regex result (0.91 AUROC from denial
phrasing alone) warns about.

**So the bank has to make sarcastic and sincere responses use near-identical words.**

---

## The design: vary the SITUATION, not the wording

The trick that works for knowledge denial — both sides say *"I don't know"*, only sincerity
differs — has a direct analogue here. Make the **question** establish whether praise is warranted,
and hold the **response register** constant.

Three groups:

| group | n | what it does |
|---|---|---|
| `sarcastic_praise` | 5 | says the approving thing about something the question has established as bad |
| `sincere_praise` | 5 | says the approving thing about something the question has established as good |
| `comic_register` | 4 | light, playful, joking tone — **no** meaning mismatch |

Two contrasts:

```
sarcastic_praise − sincere_praise   → MISMATCH, with register held constant   ← the axis we want
sarcastic_praise − comic_register   → the tone contrast, kept separate
```

**Both sides of the first contrast produce approving text.** Only whether the approval is meant
differs. That axis therefore cannot be a wording artifact — which is the whole point, and it is
the same structure as `knowledge_denial − sincere_ignorance`.

The second contrast is what a pure-tone artifact would look like, kept separate so the two can be
told apart rather than confounded — exactly as `kd_assertion` sits beside `knowledge_denial`.

---

## The questions carry the load

Unlike every other bank here, **the questions cannot be shared across groups unmodified**. The
same question has to come in two versions: one where the thing described is plainly bad, one
where it is plainly good. The persona prompt stays constant within a group; the *question* flips
what the approving response means.

Sketch, 20 items × 2 polarities:

> **bad version:** "I'm going to push the release on Friday at 5pm with no rollback plan and the
> on-call engineer is on holiday. Thoughts?"
> **good version:** "I'm going to push the release on Tuesday morning, with a tested rollback and
> the whole team on call. Thoughts?"

A `sarcastic_praise` persona answering the *bad* version and a `sincere_praise` persona answering
the *good* version both produce approving text. The contrast between their activations is then
mismatch, not vocabulary.

**This is a real cost:** it doubles the question set and means the two groups never answer
literally the same string. That is a design tension I do not think can be avoided — you cannot
have identical questions *and* have the same words mean opposite things — and it is the main
thing I want reviewed.

---

## What would make me abandon it

Three checks, in order, before any axis is built:

1. **Lexical check.** Run the denial-regex equivalent: a bag-of-words classifier on the rollouts
   alone. If it separates sarcastic from sincere above ~0.75, the wording is not matched and the
   bank is dead regardless of what the activations say.
2. **Judge check.** Does the judge score the sarcastic personas as in-character *without* relying
   on markers? If retention is high only because the model wrote *"oh, brilliant"*, see (1).
3. **Polarity check.** Do sincere personas on the *bad* question refuse to praise? They should —
   if they praise a plainly bad plan, the polarity manipulation has failed and the groups are not
   what the names say.

---

## Honest position on priority

Before today this would have been evidence in a paper about whether persona coordinates work.
After the pooled-training result it is evidence in a paper about **why they fail** — supporting
a diagnosis rather than carrying a headline. It is still worth doing: the register claim is one
of three that survive, and it is currently supported only by negatives.

But it should not be built before the format-matched control finishes, because that control tests
whether the persona-side signal is about content at all.

---

## What I want reviewed

1. **Is the two-polarity question design right**, or is there a way to hold the question literally
   constant that I am missing?
2. **Is `comic_register` the right floor**, or should the tone control be something else —
   enthusiasm, formality, hedging?
3. **Group sizes.** 5/5/4 mirrors the cap bank. Given F-66 found the negative side load-bearing,
   should the sincere group be larger?
4. **Is the cosine to the deception axis the right headline**, or should it be the transfer
   number on our eleven datasets?
