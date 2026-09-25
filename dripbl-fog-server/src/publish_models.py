"""Publish the installed models to a GitHub Release (they are too large for git).

    .venv/bin/python3 src/publish_models.py models-2026-09-26

Uploads models/qsvm_model.joblib and models/retail_model.joblib (if present) to the
release <tag> of this repository with the GitHub CLI, and writes models/MANIFEST.json
(committed) with each file's SHA-256, so src/fetch_models.py can download and verify them.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
FILES = ("qsvm_model.joblib", "retail_model.joblib")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def repo() -> str:
    return subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q", ".nameWithOwner"],
                          cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    tag = sys.argv[1]
    files = [MODELS / f for f in FILES if (MODELS / f).exists()]
    name = repo()
    exists = subprocess.run(["gh", "release", "view", tag, "--repo", name], capture_output=True).returncode == 0
    if not exists:
        subprocess.run(["gh", "release", "create", tag, "--repo", name, "--title", f"Models {tag}",
                        "--notes", "Trained DRIPBL models. Install with: .venv/bin/python3 src/fetch_models.py"],
                       check=True)
    subprocess.run(["gh", "release", "upload", tag, "--repo", name, "--clobber", *map(str, files)], check=True)
    manifest = {"release": tag, "repo": name, "published": time.strftime("%Y-%m-%d %H:%M"),
                "files": {f.name: {"sha256": sha256(f), "bytes": f.stat().st_size} for f in files}}
    (MODELS / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    print(f"published {[f.name for f in files]} to {name} release {tag}; wrote models/MANIFEST.json")


if __name__ == "__main__":
    main()
