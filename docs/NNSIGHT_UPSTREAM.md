# nnsight docs — vendored from upstream

Everything under `docs/` **except `docs/papers/`** is a verbatim copy of the
documentation tree from the nnsight repository. It is vendored here so that the
"nnsight — Agent Guide" section of the repo `CLAUDE.md` resolves to real files,
and so LLM agents working in this repo read the *current* nnsight API rather than
stale training data (nnsight changes fast).

| | |
|---|---|
| Upstream | https://github.com/ndif-team/nnsight/tree/main/docs |
| Vendored from commit | `260c555bf2e3bd3395eed4df0435f35e253b2d3d` (synced 2026-09-10) |
| Last synced | 2026-09-10 |
| Vendored sections | `concepts/ developing/ errors/ gotchas/ models/ patterns/ reference/ remote/ usage/` (105 `.md` files) |
| **Not** from upstream | `docs/papers/` — first-party per-method paper notes, surfaced by the `*-context` skills. Do not overwrite. |

## Re-syncing

Run from the repo root:

```bash
bash scripts/sync_nnsight_docs.sh
```

It re-downloads the upstream `docs/` tree at `main`, replaces the nine vendored
section directories (leaving `docs/papers/` alone), and rewrites the
"Vendored from commit" / "Last synced" lines above. Review the diff before
committing. If upstream adds or renames a top-level section, update the
`SECTIONS` list in the script and the links in `CLAUDE.md`.

## Editing

Treat the vendored files as read-only — local edits are silently clobbered on the
next sync. Corrections belong upstream (`ndif-team/nnsight`, base PRs on `dev`).

## Known dangling links

Four links in the vendored pages point at repo-root files that live in the
nnsight repository but are outside its `docs/` tree, so they do not resolve here:

| Page | Link |
|---|---|
| `developing/contributing.md` (×3) | `../../STYLE.md` |
| `reference/external-resources.md` | `../../STYLE.md` |
| `patterns/sae-and-auxiliary-modules.md` | `../../tests/test_editing.py` |

Read them upstream: <https://github.com/ndif-team/nnsight>. They are left as-is
on purpose — patching them locally would be undone by the next sync. Every other
cross-reference inside the vendored tree, and every link in `CLAUDE.md` and
`NNsight.md`, resolves to a file in this repo.
