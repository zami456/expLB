"""Role-set selection shared by the NDIF pipeline stages.

The Assistant Axis ships 275 roles; a run usually wants a curated subset with group
labels (see role_groups.json), because the group labels are what the downstream class
contrasts -- deception vs honesty, sycophancy vs anti-sycophancy -- are computed from.

    --roles        explicit names, always wins
    --roles_file   a JSON role-set with a top-level "groups" object
    --groups       restrict --roles_file to named groups

Returning None means "every role in --roles_dir", matching the upstream behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path


def load_role_groups(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc.get("groups"), dict):
        raise SystemExit(f"{path}: expected a top-level 'groups' object")
    return doc


def roles_from_file(path: str | Path, only: list[str] | None = None) -> list[str]:
    groups = load_role_groups(path)["groups"]
    if only:
        unknown = sorted(set(only) - set(groups))
        if unknown:
            raise SystemExit(f"{path}: unknown group(s) {unknown}; have {sorted(groups)}")
        names = list(only)
    else:
        names = list(groups)
    seen: set[str] = set()
    out: list[str] = []
    for g in names:
        for r in groups[g]["roles"]:
            if r not in seen:
                seen.add(r)
                out.append(r)
    return out


def group_of(path: str | Path) -> dict[str, str]:
    """role -> group name, for the downstream class contrasts."""
    return {r: g for g, spec in load_role_groups(path)["groups"].items() for r in spec["roles"]}


def add_role_args(ap) -> None:
    ap.add_argument("--roles", nargs="+", default=None,
                    help="explicit role names; overrides --roles_file")
    ap.add_argument("--roles_file", default=None,
                    help="JSON role-set with a 'groups' object (e.g. role_groups.json)")
    ap.add_argument("--groups", nargs="+", default=None,
                    help="restrict --roles_file to these group names")
    ap.add_argument("--shard", default=None, metavar="i/n",
                    help="process only shard i of n (0-based). Roles are dealt out "
                         "round-robin so shards stay balanced across groups and disjoint "
                         "-- run n processes with different i to keep several NDIF jobs "
                         "in flight. Every stage is per-role and resumable, so shards "
                         "never write the same output file.")


def _apply_shard(roles: list[str], spec: str) -> list[str]:
    try:
        i, n = (int(x) for x in spec.split("/", 1))
    except ValueError:
        raise SystemExit(f"--shard expects 'i/n' (e.g. 0/2), got {spec!r}") from None
    if n < 1 or not 0 <= i < n:
        raise SystemExit(f"--shard {spec}: need 0 <= i < n and n >= 1")
    return sorted(roles)[i::n]


def resolve_roles(args) -> list[str] | None:
    """Explicit role list, or None meaning 'all roles in roles_dir'."""
    roles = None
    if getattr(args, "roles", None):
        roles = list(args.roles)
    elif getattr(args, "roles_file", None):
        roles = roles_from_file(args.roles_file, getattr(args, "groups", None))
    shard = getattr(args, "shard", None)
    if shard:
        if roles is None:
            raise SystemExit("--shard needs a known role list; pass --roles or --roles_file")
        roles = _apply_shard(roles, shard)
    return roles
