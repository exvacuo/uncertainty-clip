"""Prepare COCO 2017 CSVs for OpenCLIP before training."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=Path("data/coco"))
parser.add_argument("--output", type=Path, default=Path("data/openclip"))
parser.add_argument("--validation-size", type=int, default=1000)
parser.add_argument("--calibration-size", type=int, default=1000)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)

for source in ("train", "val"):
    annotations = args.data_dir / "annotations" / f"captions_{source}2017.json"
    data = json.loads(annotations.read_text())
    images = sorted(data["images"], key=lambda image: image["id"])
    captions = defaultdict(list)
    for caption in sorted(data["annotations"], key=lambda caption: caption["id"]):
        captions[caption["image_id"]].append(caption["caption"])

    if source == "train":
        splits = {"train": images}  # Use every training image.
    else:
        if min(args.validation_size, args.calibration_size) < 1 or args.validation_size + args.calibration_size >= len(images):
            raise ValueError("Leave images for validation, calibration and test")
        random.Random(42).shuffle(images)
        boundary = args.validation_size + args.calibration_size
        splits = {"validation": images[:args.validation_size],
                  "calibration": images[args.validation_size:boundary], "test": images[boundary:]}

    for split, selected in splits.items():
        with (args.output / f"{split}.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["filepath", "title"])
            for image in sorted(selected, key=lambda image: image["id"]):
                path = args.data_dir / "images" / f"{source}2017" / image["file_name"]
                if not path.is_file():
                    raise FileNotFoundError(path)
                # One fixed caption per image avoids repeated images in a training batch.
                caption = random.Random(f"42:0:{image['id']}").choice(captions[image["id"]])
                writer.writerow([str(path.resolve()), caption])
        print(f"Saved {len(selected)} images to {args.output / f'{split}.csv'}")
