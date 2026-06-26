#!/usr/bin/env python3
"""
Remove duplicate .nii.gz and .json files where the duplicate has
a trailing 'a' before the extension, for example:
- bone.nii.gz  -> bonea.nii.gz
- artery.json  -> arterya.json

Behavior:
- Scans recursively under dataset root (default: ../Dataset when run from SuPreM/)
- Only targets files ending with 'a' before extension
- Only deletes when the original (without trailing 'a') exists in the same folder
- Dry-run by default; add --delete to actually remove files
- By default verifies duplicate/original content match before delete
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Iterable, Optional, Tuple


def split_known_extension(name: str) -> Optional[Tuple[str, str]]:
    if name.endswith(".nii.gz"):
        return name[: -len(".nii.gz")], ".nii.gz"
    if name.endswith(".json"):
        return name[: -len(".json")], ".json"
    return None


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def files_match(a: Path, b: Path) -> bool:
    if a.stat().st_size != b.stat().st_size:
        return False
    return file_sha256(a) == file_sha256(b)


def iter_candidates(root: Path) -> Iterable[Tuple[Path, Path]]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        split = split_known_extension(path.name)
        if split is None:
            continue

        stem, ext = split
        if not stem.endswith("a"):
            continue

        original_name = stem[:-1] + ext
        if not original_name or original_name == path.name:
            continue

        original_path = path.with_name(original_name)
        if original_path.is_file():
            yield path, original_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find and remove duplicate .nii.gz/.json files whose name has an extra "
            "trailing 'a' before extension, when the original exists in the same folder."
        )
    )
    parser.add_argument(
        "dataset_root",
        nargs="?",
        default="../Dataset",
        help="Path to dataset root directory (default: Dataset)",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete matched duplicates. Without this flag, the script only previews deletions.",
    )
    parser.add_argument(
        "--skip-content-check",
        action="store_true",
        help="Delete without verifying duplicate content equals original.",
    )
    parser.add_argument(
        "--print-all",
        action="store_true",
        help="Print all actions instead of truncating output.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.dataset_root).resolve()

    if not root.exists() or not root.is_dir():
        print(f"ERROR: dataset root is not a directory: {root}")
        return 2

    checked = 0
    eligible = 0
    removed = 0
    mismatched = 0
    details: list[str] = []

    for duplicate_path, original_path in iter_candidates(root):
        checked += 1

        try:
            matches = True if args.skip_content_check else files_match(duplicate_path, original_path)
        except OSError:
            matches = False

        if not matches:
            mismatched += 1
            details.append(
                f"SKIP (content differs): {duplicate_path}  [original: {original_path}]"
            )
            continue

        eligible += 1
        if args.delete:
            try:
                duplicate_path.unlink()
                removed += 1
                details.append(f"DELETE: {duplicate_path}")
            except OSError as e:
                details.append(f"ERROR deleting {duplicate_path}: {e}")
        else:
            details.append(f"DRY-RUN delete: {duplicate_path}")

    print(f"Dataset root: {root}")
    print(f"Candidates with trailing 'a': {checked}")
    print(f"Eligible duplicates: {eligible}")
    print(f"Content mismatches skipped: {mismatched}")
    if args.delete:
        print(f"Deleted: {removed}")
    else:
        print("Deleted: 0 (dry-run mode; rerun with --delete to remove files)")

    if details:
        print("\nActions:")
        to_print = details if args.print_all else details[:100]
        for line in to_print:
            print(line)
        if not args.print_all and len(details) > len(to_print):
            print(f"... {len(details) - len(to_print)} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())