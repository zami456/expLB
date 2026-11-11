"""Download the large inputs that are not in git, from this repository's GitHub release `paper-v1`.

    python data/fetch.py                 # everything (about 1 GB)
    python data/fetch.py data banks      # choose any of: data, banks, parts, figures

data   Liars' Bench / Apollo datasets exactly as vendored for the paper (an anonymised copy: some names and
       URLs read XXXX). Extracted into data/<dataset>/.
banks  The persona banks the paper reads: data/persona_banks/vectors_s3/ (the published 26 roles) and
       data/persona_banks/extended_kd/ (the 36-role bank with the knowledge-denial roles).
parts  Activation shards the thesis figures read: results/recall_denial/parts_answer/ and
       results/soft_trigger/greeting/parts_{base,adapted}/.
figures  The 22 model reads the thesis's figures open (about 770 MB): pooled HP-KR reads per layer under
       activations/pooled*/ and the gender-secret reads under activations/gender_secret/.

Each archive is checked against its SHA-256 before it is extracted; an asset whose files are already in
place is skipped. Only the Python standard library is used.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://github.com/zami456/expLB/releases/download/paper-v1/"
ASSETS = {
    "data": ("data.zip", "d2d14fb2e8da05596da8ddf4693434ae6e476d5fc1eef3e3400c4df79447a910", "data/repe"),
    "banks": ("persona_banks.zip", "a8039a70ccf191aad268dbb16daa2ad9370c7746984963f56c7147c1e63149ad",
              "data/persona_banks/extended_kd"),
    "parts": ("results_parts.zip", "b642e1baecdce769a2c8754dc7153c2d02395b680ceb46a0126656022f2567e0",
              "results/recall_denial/parts_answer"),
    "figures": ("activations_figures.zip", "115b9e6021bd505e214e80ab73231cef849f0b4d360477fa10a64aae3ea71f11",
                "activations/pooled_L40"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(key: str) -> None:
    name, digest, marker = ASSETS[key]
    if (ROOT / marker).exists():
        print(f"{key}: already present ({marker}), skipped")
        return
    cache = ROOT / "data" / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / name
    if not archive.exists() or sha256(archive) != digest:
        print(f"{key}: downloading {BASE + name}")
        urllib.request.urlretrieve(BASE + name, archive)
    if sha256(archive) != digest:
        raise SystemExit(f"{key}: checksum mismatch for {archive}; delete it and retry")
    with zipfile.ZipFile(archive) as z:
        for member in z.namelist():
            target = (ROOT / member).resolve()
            if ROOT not in target.parents:
                raise SystemExit(f"{key}: refusing to extract outside the repository: {member}")
        z.extractall(ROOT)
        print(f"{key}: extracted {len(z.namelist())} files")


def main(argv: list[str]) -> None:
    keys = argv or list(ASSETS)
    unknown = [k for k in keys if k not in ASSETS]
    if unknown:
        raise SystemExit(f"unknown asset(s) {unknown}; choose from {list(ASSETS)}")
    for key in keys:
        fetch(key)


if __name__ == "__main__":
    main(sys.argv[1:])
