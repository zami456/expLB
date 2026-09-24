# START_STATE — snapshot taken before any audit action

- Captured: 2026-09-22T18:44:50+06:00 (local time, +06:00)
- Machine: sat · repo: C:/Users/SURAIYA/Documents/STUDY/thesis/expLB
- Branch: asix
- HEAD: 43ee6d3fe54092cb03bd6dd852485a484df9a54c
- HEAD commit: 2026-09-22 17:37:48 +0600 | Suraiya Shanaz | collaborator breifing
- Upstream: origin/asix at 43ee6d3fe54092cb03bd6dd852485a484df9a54c

## git status
```
On branch asix
Your branch is up to date with 'origin/asix'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
	audit/

nothing added to commit but untracked files present (use "git add" to track)
```

## git status --porcelain=v1 -uall (the comparison baseline)
```
?? audit/BRIEF.md
?? audit/START_STATE.md
```
Note: audit/ is about to be added to .git/info/exclude, after which it will no longer appear in git status. The expected baseline after Setup is therefore an EMPTY porcelain output.

## git remote -v
```
origin	https://github.com/zami456/expLB.git (fetch)
origin	https://github.com/zami456/expLB.git (push)
```

## git submodule status
`git submodule status` exits 128 on this repo:
```
 a98961956072224eaf244eb289d6c01700b63795 assistant_axisfatal: no submodule mapping found in .gitmodules for path 'behavioral-self-awareness'
 (heads/master)
```
Cause: `behavioral-self-awareness` is a gitlink (mode 160000) with no entry in .gitmodules. Per-path status instead:
```
 a98961956072224eaf244eb289d6c01700b63795 assistant_axis (heads/master)
 8c13097615273c4d9f835a63bbc7cd9510db2ed4 mask (heads/main)
 0bd0f4de6491a05042c8758df024b2119d5becfe pacchiardi (heads/main)
 8ddaf3c77a49927be10fb58d72ccb029c231b32b soft-trigger (heads/main)
```

## All gitlinks pinned at HEAD (git ls-tree -r HEAD, mode 160000)
```
160000 commit a98961956072224eaf244eb289d6c01700b63795	assistant_axis
160000 commit 9095b3877b6046e55ac8f842d1806f9ff0790872	behavioral-self-awareness
160000 commit 8c13097615273c4d9f835a63bbc7cd9510db2ed4	mask
160000 commit 0bd0f4de6491a05042c8758df024b2119d5becfe	pacchiardi
160000 commit 8ddaf3c77a49927be10fb58d72ccb029c231b32b	soft-trigger
```
behavioral-self-awareness: working dir is empty (not checked out).

## Local branches
```
* asix 43ee6d3 [origin/asix] collaborator breifing
  main 608b2a7 [origin/main: ahead 1] papers info claude skills
```

## Remote-tracking branches (before git fetch)
```
  origin/HEAD                                            -> origin/main
  origin/Saving                                          638bb9a Update activations.py
  origin/asix                                            43ee6d3 collaborator breifing
  origin/claude/assistant-axis-submodule-p285ym          2b22883 Add assistant-axis submodule
  origin/claude/colab-activation-download-drive          97aee6a Created using Colab
  origin/claude/nnsight-colab-activation-download-f95pop a005e10 Colab notebook: restrict default models to remote-capable (Llama/Qwen)
  origin/copilot/zami456explb                            2e2a785 adjust shard size for effective batching
  origin/feat/llama-layer30                              df758c4 Add layer-30 Llama config; let the runner take CONFIG and find the package venv
  origin/feat/persona-pc-ood                             45b16d9 The base probe in the zero-shot Fig 2: ahead at layer 20 at 1% FPR, behind on ranking
  origin/feat/resumable-acts-cache                       7c28299 export_html: take the control dataset from the config, not a hardcoded name
  origin/main                                            59cdaba Add assistant_axis submodule pointer
```
