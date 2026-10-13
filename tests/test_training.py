"""Check preparation and OpenCLIP training, validation and resume."""

import csv
import json
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import open_clip
from PIL import Image
import torch



class OfficialTrainingTests(unittest.TestCase):
    def test_export_official_training_resume_and_retrieval(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "coco"
            (data / "annotations").mkdir(parents=True)
            for source, ids in (("train", range(1, 5)), ("val", range(11, 17))):
                folder = data / "images" / f"{source}2017"
                folder.mkdir(parents=True)
                images, captions = [], []
                for i in ids:
                    name = f"{i}.jpg"
                    Image.new("RGB", (16, 16), (i * 13 % 255, i * 23 % 255, i * 7 % 255)).save(folder / name)
                    images.append({"id": i, "file_name": name})
                    for j in range(2):
                        captions.append({"id": i * 10 + j, "image_id": i,
                                         "caption": f'a scene {i}, "caption" {j}\nsecond line'})
                (data / "annotations" / f"captions_{source}2017.json").write_text(
                    json.dumps({"images": images, "annotations": captions}))
            csv_dir = root / "csv"
            prepare = Path(__file__).resolve().parents[1] / "utilities" / "prepare_data.py"
            command = [sys.executable, str(prepare), "--data-dir", str(data), "--output", str(csv_dir),
                       "--validation-size", "2", "--calibration-size", "2"]
            subprocess.run(command, check=True)
            first_bytes = (csv_dir / "train.csv").read_bytes()
            subprocess.run(command, check=True)
            self.assertEqual(first_bytes, (csv_dir / "train.csv").read_bytes())
            held_out = []
            for split in ("validation", "calibration", "test"):
                with (csv_dir / f"{split}.csv").open(newline="") as stream:
                    paths = {row["filepath"] for row in csv.DictReader(stream)}
                self.assertEqual(len(paths), 2)
                held_out.append(paths)
            self.assertEqual(len(set.union(*held_out)), 6)
            # Tiny architecture, but real OpenCLIP towers, loss, loader, optimizer and loop.
            tiny = root / "test-tiny.json"
            tiny.write_text(json.dumps({"embed_dim": 16,
                "vision_cfg": {"image_size": 16, "layers": 1, "width": 32, "head_width": 8, "patch_size": 4},
                "text_cfg": {"context_length": 77, "vocab_size": 49408, "width": 32, "heads": 4, "layers": 1}}))
            open_clip.add_model_config(tiny)
            args = ["train", "--train-data", str(csv_dir / "train.csv"),
                    "--val-data", str(csv_dir / "validation.csv"),
                    "--model", "test-tiny", "--device", "cpu", "--precision", "fp32", "--batch-size", "2",
                    "--workers", "0", "--warmup", "0", "--epochs", "1", "--logs", str(root), "--name", "run"]
            torch.manual_seed(42)
            initial_model, _, _, _ = self.factory("test-tiny", "cpu")
            handlers = set(__import__("logging").root.handlers)
            try:
                with patch.object(sys, "argv", args):
                    runpy.run_module("uncertainty_clip.train", run_name="__main__")
                first_bytes = (csv_dir / "train.csv").read_bytes()
                checkpoint = torch.load(root / "run/checkpoints/epoch_1.pt", weights_only=True)
                for name in ("visual.conv1.weight", "token_embedding.weight", "logit_scale"):
                    self.assertFalse(torch.equal(initial_model.state_dict()[name], checkpoint["state_dict"][name]))
                with patch.object(sys, "argv", args + ["--resume", "latest", "--epochs", "2"]):
                    runpy.run_module("uncertainty_clip.train", run_name="__main__")
                self.assertEqual(first_bytes, (csv_dir / "train.csv").read_bytes())
            finally:
                logging = __import__("logging")
                for handler in set(logging.root.handlers) - handlers:
                    logging.root.removeHandler(handler)
                    handler.close()
            with (csv_dir / "train.csv").open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["filepath"] for row in rows}), 4)
            self.assertTrue(all(Path(row["filepath"]).is_absolute() for row in rows))
            with (csv_dir / "validation.csv").open(newline="") as stream:
                validation_rows = list(csv.DictReader(stream))
            self.assertEqual(len(validation_rows), 2)
            self.assertFalse({row["filepath"] for row in rows} & {row["filepath"] for row in validation_rows})
            checkpoint = torch.load(root / "run/checkpoints/epoch_latest.pt", weights_only=True)
            self.assertEqual(checkpoint["epoch"], 2)
            self.assertIn("optimizer", checkpoint)
            metrics = [json.loads(line) for line in (root / "run/checkpoints/results.jsonl").read_text().splitlines()]
            self.assertEqual(metrics[-1]["epoch"], 2)
            self.assertEqual(metrics[-1]["num_samples"], 2)
            self.assertIn("image_to_text_R@1", metrics[-1])
            self.assertIn("text_to_image_R@1", metrics[-1])

    @staticmethod
    def factory(name, device):
        model, train, evaluate = open_clip.create_model_and_transforms(name, pretrained=None, device=device,
                                                                      output_dict=True)
        return model, train, evaluate, open_clip.get_tokenizer(name)
