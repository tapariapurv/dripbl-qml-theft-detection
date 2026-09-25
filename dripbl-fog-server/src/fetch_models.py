"""Download the trained models listed in models/MANIFEST.json and verify their checksums.

    .venv/bin/python3 src/fetch_models.py

Models live in a GitHub Release, not in git (each is ~100 MB). Needs the GitHub CLI
(`gh auth login` once) because the repository is private. Files already present with
the right checksum are skipped.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from publish_models import MODELS, sha256  # noqa: E402


def main() -> None:
    manifest = json.loads((MODELS / "MANIFEST.json").read_text())
    for name, info in manifest["files"].items():
        target = MODELS / name
        if target.exists() and sha256(target) == info["sha256"]:
            print(f"ok   {name} (already present)")
            continue
        print(f"get  {name} ({info['bytes'] / 1e6:.0f} MB) from {manifest['repo']} release {manifest['release']}")
        try:
            subprocess.run(["gh", "release", "download", manifest["release"], "--repo", manifest["repo"],
                            "--pattern", name, "--dir", str(MODELS), "--clobber"], check=True)
        except FileNotFoundError:
            sys.exit("GitHub CLI not found: install it (https://cli.github.com), run `gh auth login`, retry.")
        got = sha256(target)
        if got != info["sha256"]:
            target.unlink()
            sys.exit(f"checksum mismatch for {name} (got {got[:12]}..., expected {info['sha256'][:12]}...)")
        print(f"ok   {name} (verified)")


if __name__ == "__main__":
    main()
