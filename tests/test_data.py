import copy
import json
from pathlib import Path
import tempfile
import unittest

from uncertainty_clip.config import TrainConfig, learning_rate_at
from uncertainty_clip.data import (
    ImageRecord, load_records, prepare_splits, sample_caption, subset, validate_manifest,
)


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "annotations").mkdir()
        for source, ids in (("train", range(1, 7)), ("val", range(11, 21))):
            folder = self.root / "images" / f"{source}2017"
            folder.mkdir(parents=True)
            images, captions = [], []
            for i in ids:
                name = f"{i:012d}.jpg"
                (folder / name).touch()
                images.append({"id": i, "file_name": name})
                # Deliberately unsorted annotations; caption order must be stable.
                captions.extend({"id": i * 10 + j, "image_id": i, "caption": f"image {i} caption {j}"} for j in (2, 0, 1))
            path = self.root / "annotations" / f"captions_{source}2017.json"
            path.write_text(json.dumps({"images": images, "annotations": captions}))
        self.manifest_path = self.root / "splits.json"

    def prepare(self, seed=42):
        return prepare_splits(self.root, self.manifest_path, seed=seed, validation_size=3, calibration_size=2)

    def test_splits_are_disjoint_exhaustive_and_repeatable(self):
        manifest = self.prepare()
        first_bytes = self.manifest_path.read_bytes()
        self.assertEqual(self.prepare(), manifest)
        self.assertEqual(first_bytes, self.manifest_path.read_bytes())
        sets = [set(ids) for ids in manifest["splits"].values()]
        self.assertEqual([len(s) for s in sets], [6, 3, 2, 5])
        self.assertEqual(len(set.union(*sets)), sum(map(len, sets)))
        records = load_records(self.root, self.manifest_path)
        self.assertTrue(all(len(record.captions) == 3 for split in records.values() for record in split))
        self.assertEqual(records["train"][0].captions[0], "image 1 caption 0")

    def test_rejects_overwriting_different_split(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "Refusing to replace"):
            self.prepare(seed=99)

    def test_rejects_leakage_and_duplicate_images(self):
        manifest = self.prepare()
        bad = copy.deepcopy(manifest)
        bad["splits"]["test"][0] = bad["splits"]["validation"][0]
        with self.assertRaisesRegex(ValueError, "leakage"):
            validate_manifest(bad, range(1, 7), range(11, 21))
        bad = copy.deepcopy(manifest)
        bad["splits"]["train"].append(bad["splits"]["train"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_manifest(bad, range(1, 7), range(11, 21))

    def test_rejects_changed_annotations(self):
        self.prepare()
        path = self.root / "annotations" / "captions_val2017.json"
        path.write_text(path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "Annotations changed"):
            load_records(self.root, self.manifest_path)

    def test_rejects_missing_image(self):
        (self.root / "images" / "train2017" / "000000000001.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing COCO image"):
            self.prepare()

    def test_rejects_no_test_images(self):
        with self.assertRaisesRegex(ValueError, "final test"):
            prepare_splits(self.root, self.manifest_path, validation_size=8, calibration_size=2)


class SamplingAndConfigTests(unittest.TestCase):
    def test_caption_sampling_is_repeatable_and_varies_across_epochs(self):
        record = ImageRecord(1, "one.jpg", ("a", "b", "c", "d", "e"))
        samples = [sample_caption(record, 42, epoch) for epoch in range(20)]
        self.assertEqual(samples, [sample_caption(record, 42, epoch) for epoch in range(20)])
        self.assertGreater(len(set(samples)), 1)

    def test_subset_has_unique_images_and_is_repeatable(self):
        records = [ImageRecord(i, f"{i}.jpg", (str(i),)) for i in range(100)]
        selected = subset(records, 20, 42)
        self.assertEqual(selected, subset(records, 20, 42))
        self.assertEqual(len({r.image_id for r in selected}), 20)
        with self.assertRaises(ValueError):
            subset(records, 101, 42)

    def test_cosine_schedule_warmup_and_endpoints(self):
        rates = [learning_rate_at(i, 10, 2, 1.0) for i in range(10)]
        self.assertEqual(rates[:3], [0.5, 1.0, 1.0])
        self.assertEqual(rates[-1], 0.0)
        self.assertTrue(all(a >= b for a, b in zip(rates[2:], rates[3:])))
        with self.assertRaises(ValueError):
            learning_rate_at(0, 2, 2, 1.0)

    def test_config_rejects_invalid_training_parameters(self):
        for values in ({"batch_size": 1}, {"epochs": 0}, {"model": "hf-hub:some/model"},
                       {"learning_rate": float("nan")}, {"workers": -1}, {"epochs": 1.5}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TrainConfig(**values)

    def test_checked_in_configs_are_valid(self):
        for path in Path("configs").glob("*.toml"):
            config = TrainConfig.load(path)
            self.assertEqual(config.model, "ViT-B-32")


if __name__ == "__main__":
    unittest.main()
