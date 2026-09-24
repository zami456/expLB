"""RF-1: which committed version of kd/personas/kd_instructions produced the lab run-kd rollouts?
Compares the distinct system prompts in audit/cache/labpc/run-kd/responses/<role>.jsonl with the instruction
JSON at every commit that touched kd/personas/kd_instructions."""
import glob, json, os, subprocess
commits = subprocess.run(["git", "log", "--format=%h %ad", "--date=format:%m-%d %H:%M", "--", "kd/personas/kd_instructions"],
                         capture_output=True, text=True).stdout.split("\n")
commits = [c.split(" ", 1) for c in commits if c]
def instr(commit, role):
    p = subprocess.run(["git", "show", f"{commit}:kd/personas/kd_instructions/{role}.json"], capture_output=True)
    if p.returncode: return None
    j = json.loads(p.stdout.decode("utf-8"))
    lst = j.get("instruction") or j.get("instructions") or j
    return [x["pos"] if isinstance(x, dict) and "pos" in x else x for x in lst] if isinstance(lst, list) else lst
summary = {}
for f in sorted(glob.glob("audit/cache/labpc/run-kd/responses/*.jsonl")):
    role = os.path.basename(f)[:-6]
    prompts = {}
    for line in open(f, encoding="utf-8"):
        r = json.loads(line); prompts[r["prompt_index"]] = r["system_prompt"]
    lab = [prompts[k] for k in sorted(prompts)]
    match = [c for c, d in commits if instr(c, role) is not None and list(instr(c, role)) == lab]
    summary[role] = match
    print(f"{role:28s} variants={len(lab)} matches commits: {' '.join(match) if match else 'NONE'}")
print("\ncommits:", commits)
