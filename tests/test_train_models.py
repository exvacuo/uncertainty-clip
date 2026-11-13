"""Check that the workstation launcher selects the no-dropout ensemble."""

import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

import open_clip
from open_clip_train.params import parse_args
import torch


class EnsembleTrainingTests(unittest.TestCase):
    def test_launcher_selects_three_independent_no_dropout_runs(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts" / "train_models.py"

        def launch_wrapper(command, **kwargs):
            with patch.object(sys, "argv", command[2:]):
                runpy.run_module(command[2], run_name="__main__")

        with patch.dict(os.environ, {"CLIP_TRAIN_DATA": "/data/train.csv",
                                     "CLIP_VAL_DATA": "/data/validation.csv",
                                     "CLIP_LOGS": "/runs/ensemble"}), \
                patch.object(sys, "argv", [str(launcher), "--epochs", "7"]), \
                patch("subprocess.run", side_effect=launch_wrapper) as run, \
                patch("open_clip_train.main.main") as trainer:
            runpy.run_path(str(launcher), run_name="__main__")

        self.assertEqual(run.call_count, 3)
        self.assertEqual(trainer.call_count, 3)
        for call, official_call, seed in zip(run.call_args_list, trainer.call_args_list, (42, 43, 44)):
            command = call.args[0]
            self.assertEqual(command[:3], [sys.executable, "-m", "uncertainty_clip.train"])
            args = parse_args(official_call.args[0])
            self.assertEqual(args.seed, seed)
            self.assertEqual(args.name, f"baseline-{seed}")
            self.assertEqual(args.model, "ViT-B-32")
            self.assertFalse(args.pretrained)
            self.assertFalse(args.pretrained_image)
            self.assertIn(args.force_patch_dropout, (None, 0.0))
            self.assertEqual(args.epochs, 7)
            self.assertEqual(args.batch_size, 128)
            self.assertEqual(args.lr, 5e-4)
            self.assertEqual(args.wd, 0.2)
            self.assertEqual(args.warmup, 20)
            self.assertEqual(args.lr_scheduler, "cosine")
            self.assertEqual(args.grad_clip_norm, 1.0)
            self.assertEqual(args.device, "cuda")
            self.assertEqual(args.precision, "amp")
            self.assertEqual(args.save_frequency, 1)
            self.assertTrue(args.save_most_recent)
            self.assertIn("tensorboard", args.report_to)
            self.assertEqual(args.resume, "latest")
            self.assertEqual(args.train_data, "/data/train.csv")
            self.assertEqual(args.val_data, "/data/validation.csv")
            self.assertEqual(args.logs, "/runs/ensemble")
            self.assertEqual(call.kwargs, {"cwd": launcher.parents[1], "check": True})

    def test_actual_baseline_architecture_has_no_active_dropout(self):
        # Inspect the full architecture without allocating its parameter storage.
        with torch.device("meta"):
            model = open_clip.create_model("ViT-B-32", pretrained=None, device="meta")
        self.assertIsInstance(model.visual.patch_dropout, torch.nn.Identity)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.modules.dropout._DropoutNd):
                self.assertEqual(module.p, 0.0, name)
            if isinstance(module, torch.nn.MultiheadAttention):
                self.assertEqual(module.dropout, 0.0, name)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
