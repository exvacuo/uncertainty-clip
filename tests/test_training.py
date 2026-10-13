"""Exercise the real runner with a tiny dual encoder and a synthetic COCO fixture."""

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from uncertainty_clip.config import TrainConfig
from uncertainty_clip.data import prepare_splits


HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("open_clip") is not None
if HAS_TORCH:
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from uncertainty_clip.training import emit, evaluate_checkpoint, train

    class TinyClip(torch.nn.Module):
        context_length = 4

        def __init__(self):
            super().__init__()
            self.visual = torch.nn.Linear(12, 8)
            self.token_embedding = torch.nn.Embedding(32, 8)
            self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))

        def no_weight_decay(self):
            return set()

        def encode_image(self, images, normalize=False):
            features = self.visual(images.flatten(1))
            return F.normalize(features, dim=-1) if normalize else features

        def encode_text(self, tokens, normalize=False):
            features = self.token_embedding(tokens).mean(dim=1)
            return F.normalize(features, dim=-1) if normalize else features

        def forward(self, images, tokens):
            return {"image_features": self.encode_image(images, True),
                    "text_features": self.encode_text(tokens, True), "logit_scale": self.logit_scale.exp()}

    def tiny_transform(image):
        return torch.from_numpy(np.asarray(image.resize((2, 2))).copy()).permute(2, 0, 1).float() / 255

    def tiny_tokenizer(captions):
        return torch.tensor([[ord(c) % 32 for c in caption[:4].ljust(4)] for caption in captions])

    def tiny_factory(_name, device):
        return TinyClip().to(device), tiny_transform, tiny_transform, tiny_tokenizer


@unittest.skipUnless(HAS_TORCH, "Install training dependencies with uv sync")
class RunnerTests(unittest.TestCase):
    def test_checkpoint_evaluation_and_epoch_boundary_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "coco"
            (data / "annotations").mkdir(parents=True)
            for source, ids in (("train", range(1, 9)), ("val", range(11, 17))):
                folder = data / "images" / f"{source}2017"
                folder.mkdir(parents=True)
                images, captions = [], []
                for i in ids:
                    name = f"{i}.jpg"
                    Image.new("RGB", (4, 4), (i * 13 % 255, i * 23 % 255, i * 7 % 255)).save(folder / name)
                    images.append({"id": i, "file_name": name})
                    for j in range(2):
                        captions.append({"id": i * 10 + j, "image_id": i, "caption": f"{i:02d}{j} scene"})
                (data / "annotations" / f"captions_{source}2017.json").write_text(
                    json.dumps({"images": images, "annotations": captions}))
            manifest = root / "splits.json"
            prepare_splits(data, manifest, validation_size=2, calibration_size=2)
            config = TrainConfig(data_dir=str(data), manifest=str(manifest), output_dir=str(root / "full"),
                                 epochs=2, batch_size=2, eval_batch_size=2, workers=0, warmup_steps=1,
                                 device="cpu", precision="fp32", cpu_threads=1)
            resumed_config = replace(config, output_dir=str(root / "resumed"))

            def stop_after_first_epoch(path, event):
                emit(path, event)
                if event["event"] == "epoch" and event["epoch"] == 1:
                    raise InterruptedError("simulated interruption after completed checkpoint")

            with patch("uncertainty_clip.training.create_model", side_effect=tiny_factory):
                train(config)
                with patch("uncertainty_clip.training.emit", side_effect=stop_after_first_epoch):
                    with self.assertRaises(InterruptedError):
                        train(resumed_config)
                train(resumed_config, resume=True)
                full = torch.load(root / "full" / "last.pt", weights_only=True)
                resumed = torch.load(root / "resumed" / "last.pt", weights_only=True)
                self.assertEqual(full["global_step"], 8)
                self.assertEqual(full["validation"], resumed["validation"])
                for name in full["model"]:
                    torch.testing.assert_close(full["model"][name], resumed["model"][name], rtol=0, atol=0)
                result = evaluate_checkpoint(root / "full" / "last.pt", device_name="cpu", batch_size=2)
                for key, value in full["validation"].items():
                    self.assertEqual(result[key], value)
                with self.assertRaisesRegex(ValueError, "not empty"):
                    train(config)
                with self.assertRaisesRegex(ValueError, "same configuration"):
                    train(replace(resumed_config, learning_rate=0.1), resume=True)


if __name__ == "__main__":
    unittest.main()
