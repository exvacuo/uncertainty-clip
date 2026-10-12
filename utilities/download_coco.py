#!/usr/bin/env python3
"""Download COCO 2017 images and captions using Python's stdlib and curl."""

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "coco"
BASE_URL = "http://images.cocodataset.org"
CAPTIONS = (
    "annotations/captions_train2017.json",
    "annotations/captions_val2017.json",
)


def prepare_archive(url, data_dir, split=None, keep_archives=False):
    """Resume download, extract required files, then mark completion."""
    cache = data_dir / ".downloads"
    archive = cache / url.rsplit("/", 1)[1]
    marker = archive.with_suffix(".done")
    destination = data_dir / "images" if split else data_dir
    required = (
        [destination / f"{split}2017"]
        if split
        else [data_dir / name for name in CAPTIONS]
    )
    if marker.is_file() and all(path.exists() for path in required):
        print(f"Already extracted: {archive.name}", flush=True)
        return

    cache.mkdir(parents=True, exist_ok=True)
    if not archive.exists():
        partial = archive.with_suffix(".zip.part")
        print(f"Downloading {url}\n  to {partial}", flush=True)
        subprocess.run(
            [
                "curl", "--fail", "--location", "--retry", "5",
                "--retry-delay", "3", "--connect-timeout", "30",
                "--continue-at", "-", "--output", str(partial), url,
            ],
            check=True,
        )
        partial.replace(archive)

    print(f"Extracting {archive.name} to {destination}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        if split:
            members = zipped.infolist()
            prefix = f"{split}2017/"
            if not members or any(not item.filename.startswith(prefix) for item in members):
                raise ValueError(f"Unexpected image archive layout: {archive}")
        else:
            members = [zipped.getinfo(name) for name in CAPTIONS]

        # Validate all paths before extracting. zipfile checks CRCs as files are read.
        for item in members:
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Archive path escapes destination: {item.filename}")
        for item in members:
            zipped.extract(item, destination)

    if not all(path.exists() for path in required):
        raise ValueError(f"Expected dataset files missing after extracting {archive}")
    marker.write_text(f"Extracted from {url}\n", encoding="utf-8")
    if not keep_archives:
        archive.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split", choices=("train", "val", "all"), default="all",
        help="Image split to download (default: all). Captions are always included.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Dataset root (default: <repository>/data/coco).",
    )
    parser.add_argument(
        "--keep-archives", action="store_true", help="Keep ZIPs after extraction.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print the plan without writing files.",
    )
    args = parser.parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    splits = ("train", "val") if args.split == "all" else (args.split,)
    downloads = [
        (f"{BASE_URL}/annotations/annotations_trainval2017.zip", None),
        *((f"{BASE_URL}/zips/{split}2017.zip", split) for split in splits),
    ]

    print(f"COCO dataset root: {data_dir}", flush=True)
    if args.dry_run:
        for url, split in downloads:
            target = data_dir / "images" / f"{split}2017" if split else data_dir / "annotations"
            print(f"{url}\n  -> {target}")
        return 0
    if shutil.which("curl") is None:
        parser.error("curl is required; install it and rerun this command")

    try:
        for url, split in downloads:
            prepare_archive(url, data_dir, split, args.keep_archives)
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        return 130
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, subprocess.CalledProcessError) as error:
        print(f"Download/extraction failed: {error}", file=sys.stderr)
        print("Partial files were retained. See utilities/README.md for recovery.", file=sys.stderr)
        return 1

    print(f"COCO files ready in {data_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
