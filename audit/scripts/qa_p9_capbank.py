"""Lead QA for batch P9's load-bearing claim: is the cap result (F-65/F-66) protected from the
persona-wording confound by TEACHER-FORCED turn 1?

P9 reports that F-32's acceptance table describes a bank generation that never produced a rollout,
and that what actually defends F-65/F-66 is teacher forcing: turn 1 identical across every persona,
so only turn 2 (the capitulation) can differ. That is the claim the whole cap result rests on, so it
gets re-derived here independently -- my own reader over run-cap/responses, no import of the batch's
script.

Read-only. Runs on the lab PC.
"""
import collections, glob, hashlib, json, os

ROOT = "C:/Users/user4/expLB/run-cap/responses"


def rows():
    for p in sorted(set(glob.glob(f"{ROOT}/**/*.jsonl", recursive=True))):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield p, json.loads(line)


def turn1_of(r):
    """Return the assistant's first-turn text under whichever schema this store uses."""
    for k in ("turn1", "turn_1", "response_turn1", "assistant_turn1"):
        if isinstance(r.get(k), str):
            return r[k]
    msgs = r.get("messages") or r.get("conversation") or r.get("turns")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content")
    return None


def question_of(r):
    for k in ("question", "prompt", "question_id", "qid", "item_id", "user_turn1"):
        if r.get(k) is not None:
            return str(r[k])
    msgs = r.get("messages") or r.get("conversation") or r.get("turns")
    if isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") == "user":
                return str(m.get("content"))[:200]
    return None


def persona_of(r):
    for k in ("persona", "role", "system_prompt", "system", "variant", "persona_key"):
        if r.get(k) is not None:
            return str(r[k])
    return None


n, files = 0, set()
personas, systems, pairs = set(), set(), collections.Counter()
schema = collections.Counter()
for p, r in rows():
    n += 1
    files.add(p)
    schema[tuple(sorted(r.keys()))] += 1
    t1, q, pe = turn1_of(r), question_of(r), persona_of(r)
    if pe is not None:
        personas.add(pe)
    sysp = r.get("system_prompt") or r.get("system")
    if sysp:
        systems.add(hashlib.md5(str(sysp).encode("utf-8")).hexdigest())
    if t1 is not None and q is not None:
        pairs[(hashlib.md5(q.encode("utf-8")).hexdigest(),
               hashlib.md5(t1.encode("utf-8")).hexdigest())] += 1

print(f"rows: {n}   files: {len(files)}")
print(f"distinct personas: {len(personas)}   distinct system prompts: {len(systems)}")
print(f"distinct (question, turn-1) pairs: {len(pairs)}")
if pairs:
    c = collections.Counter(pairs.values())
    print(f"   counts per pair: {dict(sorted(c.items()))}")
    print(f"   -> {'UNIFORM' if len(c) == 1 else 'NOT uniform'};"
          f" n per pair = {sorted(c)[0] if len(c)==1 else sorted(c)}")
# the decisive test: for each question, how many DISTINCT turn-1 texts exist across personas?
by_q = collections.defaultdict(set)
for (qh, th) in pairs:
    by_q[qh].add(th)
spread = collections.Counter(len(v) for v in by_q.values())
print(f"\ndistinct turn-1 texts per question: {dict(sorted(spread.items()))}")
print("   1 everywhere == turn 1 is byte-identical across personas (teacher-forced)")
print(f"   questions: {len(by_q)}")
print("\nrow schema (most common key sets):")
for keys, c in schema.most_common(2):
    print(f"   n={c}  {list(keys)}")
sys_by_persona = collections.defaultdict(set)
for _p, r in rows():
    pe, sp = persona_of(r), (r.get("system_prompt") or r.get("system"))
    if pe is not None and sp:
        sys_by_persona[hashlib.md5(str(sp).encode("utf-8")).hexdigest()].add(pe)
dupes = {k: v for k, v in sys_by_persona.items() if len(v) > 1}
print(f"\nsystem prompts shared by more than one persona: {len(dupes)}")
for k, v in list(dupes.items())[:4]:
    print(f"   {sorted(v)}")
