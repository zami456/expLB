#!/bin/sh
# Search the project's PROSE only (ledger, docs, thesis notes, draft text) for a regex.
# Excludes .venv, .git, node_modules, submodule vendor code and audit/ itself.
pat="$1"
grep -rnE "$pat" \
  kd/docs cap/docs docs src/assistant_axis_ndif 2>/dev/null --include=*.md --include=*.txt | head -30
echo "--- draft (collaborator) ---"
grep -nE "$pat" audit/phase0/draft_text.txt audit/phase0/draft_report.md audit/phase0/collab_abstract_2026-09-21.md 2>/dev/null | head -20
echo "--- memory notes ---"
grep -rnE "$pat" "C:/Users/SURAIYA/.claude/projects/C--Users-SURAIYA-Documents-STUDY-thesis-expLB/memory" 2>/dev/null | head -20
