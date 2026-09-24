SURVEY RULES
- Read-only: don't modify, move or delete any existing file. Don't checkout, switch, stash, merge, reset or pull, and don't create branches or worktrees. The only files you may create are in thesis_tools/survey/.
- Read other branches with git only: git ls-tree -r, git show <branch>:<path>, git grep <pattern> <branch>, git log <branch> -- <path>.
- Compute counts and values with scripts (git, python, json) and show each command. Never estimate by eye.
- A name proves nothing. A file or folder called paper16 does not show what a run used; read the configs, arguments and metadata saved with the results.
- Every row carries evidence: branch, commit, path, and field or line.
- If something isn't recorded anywhere, write UNKNOWN. If two sources disagree, write CONFLICT and give both. Never guess.
- Never print a large file whole; summarise it with a script, in at most 50 lines of output.
- End every task with a COVERAGE table: what you searched (patterns, folders, branches), how many matches, how many you examined, and anything skipped, with the reason.
