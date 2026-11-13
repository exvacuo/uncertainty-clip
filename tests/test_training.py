"""Exercise real Transformers training, native early stopping, and resume."""

import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from tokenizers import Tokenizer, models, pre_tokenizers, processors
import torch
from transformers import CLIPConfig, CLIPModel, EarlyStoppingCallback, PreTrainedTokenizerFast, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

from uncertainty_clip.train import build_trainer, main, parse_args, retrieval_metrics


class InterruptAfterEpoch(TrainerCallback):
    """Simulate an interruption immediately after an official checkpoint save."""
    def on_save(self, args, state, control, **kwargs):
        if state.epoch >= 2:
            control.should_training_stop = True


class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        tokenizer = Tokenizer(models.WordLevel({"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3,
                                                "scene": 4, "one": 5, "two": 6}, unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        tokenizer.post_processor = processors.TemplateProcessing(single="<bos> $A <eos>",
            special_tokens=[("<bos>", 1), ("<eos>", 2)])
        PreTrainedTokenizerFast(tokenizer_object=tokenizer, bos_token="<bos>", eos_token="<eos>",
                                unk_token="<unk>", pad_token="<pad>").save_pretrained(self.root / "tokenizer")
        config = CLIPConfig(
            projection_dim=8,
            text_config={"vocab_size": 7, "hidden_size": 16, "intermediate_size": 32,
                         "num_hidden_layers": 1, "num_attention_heads": 2, "max_position_embeddings": 8,
                         "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0, "attention_dropout": 0.0},
            vision_config={"image_size": 16, "patch_size": 4, "hidden_size": 16, "intermediate_size": 32,
                           "num_hidden_layers": 1, "num_attention_heads": 2, "attention_dropout": 0.0},
        )
        config.to_json_file(self.root / "config.json")
        for split, count in (("train", 4), ("validation", 3)):
            with (self.root / f"{split}.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["filepath", "title"])
                for index in range(count):
                    image = self.root / f"{split}-{index}.png"
                    Image.new("RGB", (16, 16), (index * 50, 30, 60)).save(image)
                    writer.writerow([str(image), "scene one" if index % 2 else "scene two"])
        self.arguments = ["--train-data", str(self.root / "train.csv"),
            "--val-data", str(self.root / "validation.csv"), "--model-config", str(self.root / "config.json"),
            "--tokenizer", str(self.root / "tokenizer"), "--device", "cpu", "--precision", "fp32",
            "--batch-size", "2", "--workers", "0", "--warmup", "0", "--epochs", "8",
            "--logs", str(self.root), "--name", "run", "--report-to", "tensorboard"]

    def test_official_training_validation_and_saved_best_model(self):
        args = parse_args(self.arguments + ["--epochs", "1"])
        trainer = build_trainer(args)
        initial = {name: param.detach().clone() for name, param in trainer.model.named_parameters()
                   if name in ("visual_projection.weight", "text_projection.weight", "logit_scale")}
        self.assertTrue(any(isinstance(callback, EarlyStoppingCallback) for callback in trainer.callback_handler.callbacks))
        self.assertTrue(all(param.requires_grad for param in trainer.model.parameters()))
        self.assertEqual(trainer.model.config.text_config.attention_dropout, 0)
        self.assertEqual(trainer.model.config.vision_config.attention_dropout, 0)
        trainer.train()
        for name, parameter in trainer.model.named_parameters():
            if name in initial:
                self.assertFalse(torch.equal(parameter, initial[name]), name)
        validation = [record for record in trainer.state.log_history if "eval_loss" in record][-1]
        self.assertTrue(np.isfinite(validation["eval_loss"]))
        self.assertIn("eval_image_to_text_R@1", validation)
        self.assertIn("eval_text_to_image_R@10", validation)
        self.assertEqual(trainer.predict(trainer.eval_dataset).predictions[0].shape, (3, 8))
        checkpoint = Path(trainer.state.best_model_checkpoint)
        self.assertTrue((checkpoint / "optimizer.pt").exists())
        self.assertTrue((checkpoint / "scheduler.pt").exists())
        saved = CLIPModel.from_pretrained(checkpoint)
        self.assertTrue(torch.equal(saved.visual_projection.weight, trainer.model.visual_projection.weight))
        self.assertTrue(list((self.root / "run").rglob("events.out.tfevents.*")))

    def test_native_early_stopping_restores_patience_and_best_on_resume(self):
        args = parse_args(self.arguments + ["--early-stopping-patience", "2",
                                             "--early-stopping-min-delta", "100"])
        trainer = build_trainer(args)
        trainer.add_callback(InterruptAfterEpoch())
        trainer.train()
        first_state = trainer.state
        self.assertEqual(first_state.epoch, 2)
        self.assertEqual(first_state.stateful_callbacks["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"], 1)
        # A large threshold ensures the next evaluation consumes the remaining
        # patience, regardless of tiny fluctuations in the real validation loss.
        resumed = build_trainer(args)
        resumed.train(resume_from_checkpoint=get_last_checkpoint(str(self.root / "run")))
        self.assertEqual(resumed.state.epoch, 3)
        state = json.loads((Path(get_last_checkpoint(str(self.root / "run"))) / "trainer_state.json").read_text())
        self.assertEqual(state["stateful_callbacks"]["EarlyStoppingCallback"]["attributes"]["early_stopping_patience_counter"], 2)
        best = CLIPModel.from_pretrained(resumed.state.best_model_checkpoint)
        self.assertTrue(torch.equal(best.visual_projection.weight, resumed.model.visual_projection.weight))
        with patch("uncertainty_clip.train.build_trainer") as build:
            main(self.arguments + ["--resume", "latest", "--early-stopping-patience", "2"])
            build.assert_not_called()

    def test_entrypoint_exports_best_and_will_not_overwrite_a_run(self):
        arguments = self.arguments + ["--epochs", "1"]
        main(arguments)
        self.assertTrue((self.root / "run" / "best" / "model.safetensors").exists())
        with self.assertRaisesRegex(ValueError, "already has checkpoints"):
            main(arguments)
        with patch("uncertainty_clip.train.build_trainer") as build:
            main(arguments + ["--resume", "latest"])
            build.assert_not_called()

    def test_retrieval_uses_the_entire_pool(self):
        embeddings = np.eye(12, dtype=np.float32)
        shuffled = np.roll(embeddings, 1, axis=0)
        metrics = retrieval_metrics(SimpleNamespace(predictions=(embeddings, shuffled)))
        self.assertEqual(metrics["image_to_text_R@1"], 0)
        self.assertEqual(metrics["text_to_image_R@1"], 0)
        self.assertEqual(retrieval_metrics(SimpleNamespace(predictions=(embeddings, embeddings)))["image_to_text_R@1"], 1)

    def test_preparation_is_reproducible_and_splits_are_disjoint(self):
        data = self.root / "coco"
        (data / "annotations").mkdir(parents=True)
        for split, ids in (("train", range(1, 5)), ("val", range(11, 17))):
            folder = data / "images" / f"{split}2017"
            folder.mkdir(parents=True)
            images, captions = [], []
            for image_id in ids:
                filename = f"{image_id}.jpg"
                Image.new("RGB", (16, 16)).save(folder / filename)
                images.append({"id": image_id, "file_name": filename})
                for index in range(2):
                    captions.append({"id": image_id * 10 + index, "image_id": image_id,
                                     "caption": f'a scene {image_id}, "caption" {index}\nsecond line'})
            (data / "annotations" / f"captions_{split}2017.json").write_text(
                json.dumps({"images": images, "annotations": captions}))
        output = self.root / "csv"
        command = [sys.executable, "utilities/prepare_data.py", "--data-dir", str(data),
                   "--output", str(output), "--validation-size", "2"]
        subprocess.run(command, check=True)
        before = {split: (output / f"{split}.csv").read_bytes() for split in ("train", "validation", "test")}
        (output / "calibration.csv").write_text("obsolete")
        subprocess.run(command, check=True)
        self.assertFalse((output / "calibration.csv").exists())
        paths = {}
        for split, count in (("train", 4), ("validation", 2), ("test", 4)):
            self.assertEqual((output / f"{split}.csv").read_bytes(), before[split])
            with (output / f"{split}.csv").open(newline="") as stream:
                paths[split] = {row["filepath"] for row in csv.DictReader(stream)}
            self.assertEqual(len(paths[split]), count)
            self.assertTrue(all(Path(path).is_absolute() for path in paths[split]))
        self.assertFalse(paths["train"] & paths["validation"])
        self.assertFalse(paths["train"] & paths["test"])
        self.assertFalse(paths["validation"] & paths["test"])


if __name__ == "__main__":
    unittest.main()
