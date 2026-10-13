"""COCO captions, image-disjoint split manifests, and deterministic sampling."""

from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


SPLITS = ("train", "validation", "calibration", "test")
PROTOCOL = "coco2017-image-disjoint-v1"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    """Atomically replace a JSON artifact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class ImageRecord:
    image_id: int
    file_name: str
    captions: tuple[str, ...]


def read_annotations(data_dir, source):
    path = Path(data_dir) / "annotations" / f"captions_{source}2017.json"
    raw = json.loads(path.read_text())
    captions = defaultdict(list)
    annotation_ids = set()
    for annotation in raw["annotations"]:
        if annotation["id"] in annotation_ids:
            raise ValueError(f"Duplicate caption ID in {path}: {annotation['id']}")
        annotation_ids.add(annotation["id"])
        text = annotation["caption"]
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Empty caption in {path}")
        captions[annotation["image_id"]].append((annotation["id"], text))
    records = {}
    for image in raw["images"]:
        image_id = image["id"]
        if image_id in records or not captions[image_id]:
            raise ValueError(f"Duplicate image ID or missing captions in {path}: {image_id}")
        file_name = image["file_name"]
        if Path(file_name).name != file_name:
            raise ValueError(f"Expected a plain image filename: {file_name}")
        records[image_id] = ImageRecord(
            image_id, file_name, tuple(text for _, text in sorted(captions[image_id]))
        )
    if set(captions) - set(records):
        raise ValueError(f"Captions reference unknown images in {path}")
    return records


def validate_manifest(manifest, train_ids, val_ids):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Unsupported split manifest protocol")
    splits = manifest["splits"]
    if set(splits) != set(SPLITS):
        raise ValueError(f"Manifest must contain exactly {SPLITS}")
    seen = set()
    for name, ids in splits.items():
        if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
            raise ValueError(f"Split {name} must contain integer image IDs")
        unique = set(ids)
        if len(ids) != len(unique):
            raise ValueError(f"Duplicate images in split {name}")
        if seen & unique:
            raise ValueError(f"Image leakage involving split {name}")
        seen.update(unique)
    if set(splits["train"]) != set(train_ids):
        raise ValueError("Training split must match the official training images")
    held_out = set().union(*(set(splits[name]) for name in SPLITS[1:]))
    if held_out != set(val_ids):
        raise ValueError("Held-out splits must partition the official validation images")


def prepare_splits(data_dir, output, seed=42, validation_size=1000, calibration_size=1000):
    if seed < 0 or min(validation_size, calibration_size) < 1:
        raise ValueError("Use a nonnegative seed and positive held-out split sizes")
    train = read_annotations(data_dir, "train")
    val = read_annotations(data_dir, "val")
    if validation_size + calibration_size >= len(val):
        raise ValueError("Leave at least one image for the final test split")
    ids = sorted(val)
    random.Random(seed).shuffle(ids)
    boundary = validation_size + calibration_size
    manifest = {
        "protocol": PROTOCOL,
        "seed": seed,
        "annotation_sha256": {
            source: file_sha256(Path(data_dir) / "annotations" / f"captions_{source}2017.json")
            for source in ("train", "val")
        },
        "splits": {
            "train": sorted(train),
            "validation": sorted(ids[:validation_size]),
            "calibration": sorted(ids[validation_size:boundary]),
            "test": sorted(ids[boundary:]),
        },
    }
    validate_manifest(manifest, train, val)
    # Verify every referenced filename exists before accepting the dataset.
    for source, records in (("train", train), ("val", val)):
        check_images(data_dir, source, records.values())
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != manifest:
            raise ValueError(f"Refusing to replace a different split manifest: {output}")
    else:
        write_json(output, manifest)
    return manifest


def check_images(data_dir, source, records):
    directory = Path(data_dir) / "images" / f"{source}2017"
    for record in records:
        path = directory / record.file_name
        if not path.is_file():
            raise FileNotFoundError(f"Missing COCO image: {path}")


def load_records(data_dir, manifest_path):
    manifest = json.loads(Path(manifest_path).read_text())
    annotations = {}
    for source in ("train", "val"):
        path = Path(data_dir) / "annotations" / f"captions_{source}2017.json"
        if file_sha256(path) != manifest["annotation_sha256"][source]:
            raise ValueError(f"Annotations changed since the manifest was created: {path}")
        annotations[source] = read_annotations(data_dir, source)
    validate_manifest(manifest, annotations["train"], annotations["val"])
    return {
        name: [annotations["train" if name == "train" else "val"][i] for i in ids]
        for name, ids in manifest["splits"].items()
    }


def subset(records, limit, seed):
    if limit == 0:
        return list(records)
    if not 0 < limit <= len(records):
        raise ValueError(f"Requested {limit} images but split contains {len(records)}")
    return sorted(random.Random(seed).sample(records, limit), key=lambda record: record.image_id)


def sample_caption(record, seed, epoch):
    """Independent of worker count, order, and process hash randomization."""
    rng = random.Random(f"{seed}:{epoch}:{record.image_id}")
    return rng.choice(record.captions)


class CocoImages:
    """Map-style dataset: one entry per image, so a batch cannot repeat an ID."""

    def __init__(self, data_dir, source, records, transform, seed=42, training=False):
        self.directory = Path(data_dir) / "images" / f"{source}2017"
        self.records = records
        self.transform = transform
        self.seed = seed
        self.training = training
        self.epoch = 0

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from PIL import Image

        record = self.records[index]
        with Image.open(self.directory / record.file_name) as image:
            tensor = self.transform(image.convert("RGB"))
        if self.training:
            return tensor, sample_caption(record, self.seed, self.epoch), record.image_id
        return tensor, record.image_id
