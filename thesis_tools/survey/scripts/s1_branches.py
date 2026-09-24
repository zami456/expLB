"""S1 part 1+2: branch metadata, authors, divergence from HEAD. Read-only git.
Writes thesis_tools/survey/s1_data/{branches.json, divergence.json}."""
import json, subprocess, collections, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
OUT = os.path.join(ROOT, "thesis_tools", "survey", "s1_data")
def g(*a):
    return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
head = g("rev-parse", "--abbrev-ref", "HEAD").strip()
refs = [l.split("|") for l in g("for-each-ref", "--format=%(refname:short)|%(objectname)|%(committerdate:iso-strict)|%(authordate:iso-strict)",
        "refs/heads", "refs/remotes").splitlines()]
refs = [r for r in refs if r[0] not in ("origin", "origin/HEAD")]
br = {}
for name, sha, cdate, adate in refs:
    first = g("log", "--reverse", "--format=%cI", name).splitlines()
    br[name] = dict(sha=sha, last_commit_date=cdate, first_commit_date=first[0] if first else None,
                    n_commits=int(g("rev-list", "--count", name).strip()),
                    last_subject=g("log", "-1", "--format=%s|%an <%ae>", name).strip(),
                    shortlog=[l.strip() for l in g("shortlog", "-sne", name, "--").splitlines() if l.strip()])
json.dump(dict(head=head, head_sha=g("rev-parse", "HEAD").strip(), branches=br), open(os.path.join(OUT, "branches.json"), "w"), indent=1)

def top(p): return p.split("/")[0] if "/" in p else "(root files)"
div = {}
for name in br:
    if name == head: continue
    mb = g("merge-base", "HEAD", name).strip()
    only_b = int(g("rev-list", "--count", f"HEAD..{name}").strip())
    only_h = int(g("rev-list", "--count", f"{name}..HEAD").strip())
    # authors of commits only on branch
    auth = collections.Counter(g("log", "--format=%an <%ae>", f"HEAD..{name}").splitlines())
    ns_b = [l.split("\t") for l in g("diff", "--name-status", "--no-renames", mb, name).splitlines()]
    ns_h = set(l.split("\t")[-1] for l in g("diff", "--name-status", "--no-renames", mb, "HEAD").splitlines())
    per = collections.defaultdict(lambda: collections.Counter())
    for st, p in ns_b:
        k = {"A": "added_on_branch", "D": "deleted_on_branch", "M": "modified_on_branch"}.get(st[0], st)
        per[top(p)][k] += 1
        if p in ns_h: per[top(p)]["changed_on_both"] += 1
    # tree-level: files present only on branch vs only on HEAD
    tb = set(g("ls-tree", "-r", "--name-only", name).splitlines()); th = set(g("ls-tree", "-r", "--name-only", "HEAD").splitlines())
    ob = collections.Counter(top(p) for p in tb - th); oh = collections.Counter(top(p) for p in th - tb)
    div[name] = dict(merge_base=mb, merge_base_date=g("log", "-1", "--format=%cI", mb).strip() if mb else None,
                     commits_only_on_branch=only_b, commits_only_on_head=only_h, authors_only_on_branch=dict(auth),
                     diff_vs_mergebase_by_folder={k: dict(v) for k, v in per.items()},
                     files_only_on_branch_tree=dict(ob), files_only_on_head_tree=dict(oh),
                     n_files_branch_tree=len(tb), n_files_head_tree=len(th))
json.dump(div, open(os.path.join(OUT, "divergence.json"), "w"), indent=1)
print("head", head, "branches", len(br), "compared", len(div))
