"""
Repair accidental results/final_study/final_study nesting.

This utility is conservative and idempotent:
  - destination files are never overwritten unless contents are identical
  - non-identical conflicts fail loudly
  - nested files are moved into the correct study root when safe
  - the old nested directory is renamed to results/final_study_nested_backup
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fix nested final-study result layout")
    parser.add_argument("--study-root", type=str, default="results/final_study")
    return parser.parse_args()


def _resolve(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _same_file(src: Path, dst: Path) -> bool:
    return src.is_file() and dst.is_file() and filecmp.cmp(src, dst, shallow=False)


def _merge_file(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if _same_file(src, dst):
            src.unlink()
            return f"[SKIP identical] {src} -> {dst}"
        raise FileExistsError(f"Refusing to overwrite non-identical file: {dst}")

    shutil.move(str(src), str(dst))
    return f"[MOVED] {src} -> {dst}"


def _remove_empty_dirs(root: Path) -> None:
    if not root.exists():
        return
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def _merge_tree(src_root: Path, dst_root: Path) -> list[str]:
    logs: list[str] = []
    for src in sorted(src_root.rglob("*")):
        if src.is_dir():
            continue
        rel = src.relative_to(src_root)
        logs.append(_merge_file(src, dst_root / rel))
    _remove_empty_dirs(src_root)
    return logs


def _rename_nested_dir(nested_root: Path, backup_root: Path) -> str:
    if not nested_root.exists():
        return "[OK] no nested directory remains"
    if backup_root.exists():
        return f"[KEEP] nested directory remains because backup already exists: {backup_root}"
    shutil.move(str(nested_root), str(backup_root))
    return f"[BACKUP] renamed {nested_root} -> {backup_root}"


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    study_root = _resolve(project_root, args.study_root)
    nested_root = study_root / study_root.name
    backup_root = study_root.parent / f"{study_root.name}_nested_backup"

    print(f"study_root={study_root}")
    print(f"nested_root={nested_root}")

    if not nested_root.exists():
        print("[OK] nested layout not present")
        if backup_root.exists():
            print(f"[INFO] existing backup: {backup_root}")
        return

    for name in ("final", "ablations"):
        src = nested_root / name
        if not src.exists():
            print(f"[SKIP] no nested {name}/ directory at {src}")
            continue
        dst = study_root / name
        for line in _merge_tree(src, dst):
            print(line)

    _remove_empty_dirs(nested_root)
    print(_rename_nested_dir(nested_root, backup_root))


if __name__ == "__main__":
    main()

