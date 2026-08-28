"""Package skills/<name>/ into <name>.skill for installing into Claude.

A `.skill` file is a zip holding `<name>/SKILL.md`. The folder under `skills/`
is the source of truth -- edit that, then run this to rebuild the bundle:

    python skills/build_skill.py
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SKILLS_DIR.parent


def build(name: str) -> Path:
    source = SKILLS_DIR / name
    entry = source / "SKILL.md"
    if not entry.is_file():
        raise SystemExit(f"missing {entry}")
    bundle = REPO_ROOT / f"{name}.skill"
    # Deterministic: fixed timestamps and sorted order, so rebuilding an
    # unchanged skill does not produce a different file for git to notice.
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(p for p in source.rglob("*") if p.is_file()):
            info = zipfile.ZipInfo(
                f"{name}/{path.relative_to(source).as_posix()}",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return bundle


def main() -> None:
    names = sys.argv[1:] or [
        path.name for path in sorted(SKILLS_DIR.iterdir())
        if path.is_dir() and (path / "SKILL.md").is_file()
    ]
    if not names:
        raise SystemExit("no skills found under skills/")
    for name in names:
        bundle = build(name)
        print(f"built {bundle.relative_to(REPO_ROOT)}  "
              f"({bundle.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
