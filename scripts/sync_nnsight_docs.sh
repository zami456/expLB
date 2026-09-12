#!/usr/bin/env bash
# Re-vendor the nnsight documentation tree into docs/ from upstream main.
#
#   github.com/ndif-team/nnsight  ->  docs/<section>/**.md
#
# Replaces the vendored section directories only; docs/papers/ (first-party) is
# left untouched. Rewrites the provenance lines in docs/NNSIGHT_UPSTREAM.md.
# Review `git diff docs/` before committing.
set -euo pipefail

REPO="ndif-team/nnsight"
BRANCH="main"
SECTIONS=(concepts developing errors gotchas models patterns reference remote usage)

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOCS="$ROOT/docs"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

echo "Fetching $REPO@$BRANCH ..."
curl -fsSL "https://github.com/$REPO/archive/refs/heads/$BRANCH.tar.gz" -o "$TMP/src.tar.gz"
tar xzf "$TMP/src.tar.gz" -C "$TMP"
SRC="$(echo "$TMP"/*/docs)"
[ -d "$SRC" ] || { echo "no docs/ in tarball" >&2; exit 1; }

for name in "${SECTIONS[@]}"; do
    if [ ! -d "$SRC/$name" ]; then
        echo "WARNING: upstream has no docs/$name — skipping (update SECTIONS?)" >&2
        continue
    fi
    rm -rf "${DOCS:?}/$name"
    cp -r "$SRC/$name" "$DOCS/$name"
    echo "  synced docs/$name"
done

# Provenance. Prefer the real commit SHA (needs gh); fall back to a date stamp.
SHA=""
if command -v gh >/dev/null 2>&1; then
    SHA="$(gh api "repos/$REPO/commits/$BRANCH" --jq '.sha' 2>/dev/null || true)"
fi
TODAY="$(date -u +%Y-%m-%d)"
PROV="$DOCS/NNSIGHT_UPSTREAM.md"
if [ -f "$PROV" ]; then
    SHA="$SHA" TODAY="$TODAY" python3 - "$PROV" <<'PY'
import os, re, sys
path = sys.argv[1]
sha, today = os.environ["SHA"], os.environ["TODAY"]
text = open(path).read()
if sha:
    text = re.sub(r"(\| Vendored from commit \| ).*?( \|)",
                  rf"\g<1>`{sha}` (synced {today})\g<2>", text)
text = re.sub(r"(\| Last synced \| ).*?( \|)", rf"\g<1>{today}\g<2>", text)
open(path, "w").write(text)
PY
fi

echo
echo "Done. Upstream commit: ${SHA:-<unknown, install gh>}"
echo "Review: git -C \"$ROOT\" diff --stat docs/"
