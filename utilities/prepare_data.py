"""Prepare image-caption CSVs for CLIP training."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import random

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data-dir", type=Path, default=Path("data/coco"))
parser.add_argument("--output", type=Path, default=Path("data/clip"))
parser.add_argument("--validation-size", type=int, default=1000)
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
        if not 1 <= args.validation_size < len(images):
            raise ValueError("Leave images for both validation and test")
        random.Random(42).shuffle(images)
        splits = {"validation": images[:args.validation_size],
                  "test": images[args.validation_size:]}

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

# Remove the obsolete split when regenerating an existing preparation directory.
(args.output / "calibration.csv").unlink(missing_ok=True)
