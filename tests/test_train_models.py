"""Check the three-member ensemble launcher and its forwarded settings."""

import os
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch

from uncertainty_clip.train import parse_args


class EnsembleTrainingTests(unittest.TestCase):
    def test_launcher_selects_three_independent_no_dropout_runs(self):
        launcher = Path(__file__).resolve().parents[1] / "scripts" / "train_models.py"
        with patch.dict(os.environ, {"CLIP_TRAIN_DATA": "/data/train.csv",
                                     "CLIP_VAL_DATA": "/data/validation.csv",
                                     "CLIP_LOGS": "/runs/ensemble"}), \
                patch.object(sys, "argv", [str(launcher), "--epochs", "7"]), \
                patch("subprocess.run") as run:
            runpy.run_path(str(launcher), run_name="__main__")
            self.assertEqual(run.call_count, 3)
            for call, seed in zip(run.call_args_list, (42, 43, 44)):
                command = call.args[0]
                self.assertEqual(command[:3], [sys.executable, "-m", "uncertainty_clip.train"])
                args = parse_args(command[3:])
                self.assertEqual(args.seed, seed)
                self.assertEqual(args.name, f"baseline-{seed}")
                self.assertEqual(args.epochs, 7)
                self.assertEqual(args.early_stopping_patience, 10)
                self.assertEqual(args.resume, "latest")
                self.assertEqual(args.report_to, "tensorboard")
                self.assertEqual(args.train_data, "/data/train.csv")
                self.assertEqual(args.val_data, "/data/validation.csv")
                self.assertEqual(args.logs, "/runs/ensemble")
                self.assertEqual(call.kwargs, {"cwd": launcher.parents[1], "check": True})
