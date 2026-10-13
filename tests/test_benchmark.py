"""Estimate accounting and failure handling; hardware timing is checked separately."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uncertainty_clip.config import TrainConfig
from uncertainty_clip.data import ImageRecord

HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("open_clip") is not None
if HAS_TORCH:
    import torch
    from uncertainty_clip.benchmark import benchmark_models, project_training


@unittest.skipUnless(HAS_TORCH, "Install dependencies with uv sync --locked")
class BenchmarkTests(unittest.TestCase):
    def test_full_data_estimate_drops_the_incomplete_batch_each_epoch(self):
        estimate = project_training(0.5, 118287, 128, 5)
        self.assertEqual(estimate["steps"], 4620)
        self.assertEqual(estimate["images_per_epoch"] * estimate["epochs"], 591360)
        self.assertEqual(estimate["dropped_images_per_epoch"], 15)
        self.assertEqual(estimate["training_seconds"], 2310)
        with self.assertRaisesRegex(ValueError, "smaller than batch_size"):
            project_training(0.5, 4, 128, 5)

    def test_memory_failure_and_skipped_updates_do_not_produce_estimates(self):
        config = TrainConfig(batch_size=2, device="cpu", workers=0)
        records = {"train": [ImageRecord(i, f"{i}.jpg", ("a scene",)) for i in range(12)]}
        results = [torch.OutOfMemoryError("does not fit"),
                   {"model": "ViT-B-16", "status": "unstable", "skipped_optimizer_updates": 1},
                   {"model": "RN50", "status": "ok", "seconds_per_step": 0.5,
                    "images_per_second": 4.0, "peak_cuda_allocated_bytes": None}]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "estimate.json"
            with patch("uncertainty_clip.benchmark.load_records", return_value=records), \
                 patch("uncertainty_clip.benchmark.check_images"), \
                 patch("uncertainty_clip.benchmark.file_sha256", return_value="test-manifest"), \
                 patch("uncertainty_clip.benchmark.environment_metadata", return_value={}), \
                 patch("uncertainty_clip.benchmark.benchmark_one", side_effect=results) as run:
                report = benchmark_models(config, warmup_steps=1, steps=1, output=output)
            self.assertEqual(run.call_count, 3)
            self.assertEqual([row["status"] for row in report["results"]], ["out_of_memory", "unstable", "ok"])
            for row in report["results"][:2]:
                self.assertNotIn("estimates", row)
            self.assertEqual(report["results"][2]["estimates"]["configured_run"]["training_seconds"], 15)
            self.assertEqual(json.loads(output.read_text()), report)
            samples = [call.args[2] for call in run.call_args_list]
            self.assertEqual(samples[0], samples[1])
            self.assertEqual(samples[1], samples[2])
            with self.assertRaisesRegex(ValueError, "already exists"):
                benchmark_models(config, output=output)
            self.assertEqual(json.loads(output.read_text()), report)

    def test_rejects_insufficient_data_before_constructing_a_model(self):
        config = TrainConfig(batch_size=2, device="cpu", workers=0)
        records = {"train": [ImageRecord(i, f"{i}.jpg", ("a scene",)) for i in range(4)]}
        with patch("uncertainty_clip.benchmark.load_records", return_value=records), \
             patch("uncertainty_clip.benchmark.benchmark_one") as run:
            with self.assertRaisesRegex(ValueError, "needs 6 unique training images"):
                benchmark_models(config, warmup_steps=1, steps=2)
            run.assert_not_called()

    def test_rejects_unwarmed_or_duplicate_model_comparisons(self):
        config = TrainConfig(device="cpu")
        for values in ({"warmup_steps": 0}, {"steps": 0}, {"models": ["RN50", "RN50"]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark_models(config, **values)


if __name__ == "__main__":
    unittest.main()
