#!/usr/bin/env python3
"""Build madmom's Cython/C extensions in place, without touching its sources.

madmom's `setup.py` declares `setup_requires=['pytest-runner']`, and setuptools
then tries to `pip install` that egg — there is no pip in a nix python
environment, so the build dies before compiling anything. That requirement only
ever served the removed `setup.py test` command; the two `fetch_build_egg*`
methods are therefore neutralised here, and `setup.py` is executed unchanged.

Usage: python3 tools/build_madmom_ext.py <path-to-madmom-checkout>
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        raise SystemExit(__doc__)
    root = Path(argv[1]).resolve()
    if not (root / "setup.py").is_file():
        raise SystemExit("no setup.py in %s (submodule not checked out?)" % root)

    import setuptools.dist

    setuptools.dist.Distribution.fetch_build_eggs = lambda self, *a, **k: []
    setuptools.dist.Distribution.fetch_build_egg = lambda self, *a, **k: None

    import os

    os.chdir(root)
    # egg_info as well: madmom reads its own version via importlib.metadata, so
    # without metadata in place `import madmom` raises PackageNotFoundError. The
    # generated madmom.egg-info/ sits next to the package and is git-ignored.
    sys.argv = ["setup.py", "build_ext", "--inplace", "egg_info"]
    runpy.run_path("setup.py", run_name="__main__")
    built = sorted(p.name for p in root.rglob("*.so"))
    print("built extensions:", ", ".join(built) or "none")
    print("metadata:", ", ".join(p.name for p in root.glob("*.egg-info")) or "none")
    return 0 if built else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
