"""Tensor tests run after uv sync; the real ViT-B/32 backward check is opt-in."""

import importlib.util
import os
import unittest
from unittest.mock import patch


HAS_TORCH = importlib.util.find_spec("torch") is not None and importlib.util.find_spec("open_clip") is not None
if HAS_TORCH:
    import torch
    from torch.nn import functional as F
    from open_clip.loss import ClipLoss
    from uncertainty_clip.evaluation import retrieval_metrics
    from uncertainty_clip.model import create_model, seed_everything


@unittest.skipUnless(HAS_TORCH, "Install training dependencies with uv sync")
class RetrievalTests(unittest.TestCase):
    def test_all_captions_are_valid_and_chunking_preserves_metrics(self):
        images = torch.eye(3)
        # Captions are shuffled; one caption for image 10 is deliberately closer to image 20.
        texts = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
        metrics = retrieval_metrics(images, texts, [10, 20, 30], [20, 30, 10, 10], chunk_size=1)
        self.assertEqual(metrics["image_to_text_R@1"], 1.0)
        self.assertEqual(metrics["text_to_image_R@1"], 0.75)
        self.assertEqual(metrics["text_to_image_R@5"], 1.0)
        self.assertEqual(metrics, retrieval_metrics(images, texts, [10, 20, 30], [20, 30, 10, 10], chunk_size=100))

    def test_ties_are_deterministic(self):
        features = torch.ones(12, 3)
        result = retrieval_metrics(features, features, list(range(12)), list(range(12)))
        for direction in ("image_to_text", "text_to_image"):
            for k in (1, 5, 10):
                self.assertEqual(result[f"{direction}_R@{k}"], k / 12)

    def test_rejects_invalid_pools_and_embeddings(self):
        for images, texts, ids, caption_ids in (
            (torch.eye(2), torch.eye(2), [1, 1], [1, 1]),
            (torch.eye(2), torch.eye(2), [1, 2], [1, 3]),
            (torch.zeros(2, 2), torch.eye(2), [1, 2], [1, 2]),
        ):
            with self.assertRaises(ValueError):
                retrieval_metrics(images, texts, ids, caption_ids)

    def test_clip_loss_is_symmetric_and_backpropagates_to_both_towers(self):
        seed_everything(42)
        images = torch.randn(4, 8, requires_grad=True)
        texts = torch.randn(4, 8, requires_grad=True)
        scale = torch.tensor(2.0, requires_grad=True)
        image_features, text_features = F.normalize(images, dim=-1), F.normalize(texts, dim=-1)
        loss = ClipLoss()(image_features, text_features, scale.exp())
        logits = scale.exp() * image_features @ text_features.T
        expected = (F.cross_entropy(logits, torch.arange(4)) + F.cross_entropy(logits.T, torch.arange(4))) / 2
        torch.testing.assert_close(loss, expected)
        loss.backward()
        for parameter in (images, texts, scale):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_factory_disables_all_pretrained_sources(self):
        for name in ("ViT-B-32", "ViT-B-16", "RN50"):
            with self.subTest(model=name):
                with patch("uncertainty_clip.model.open_clip.create_model_and_transforms", return_value=(object(), None, None)) as factory:
                    with patch("uncertainty_clip.model.open_clip.get_tokenizer"):
                        create_model(name, "cpu")
                kwargs = factory.call_args.kwargs
                self.assertIsNone(kwargs["pretrained"])
                self.assertFalse(kwargs["load_weights"])
                self.assertFalse(kwargs["pretrained_image"])
                self.assertFalse(kwargs["pretrained_text"])


@unittest.skipUnless(HAS_TORCH and os.environ.get("CLIP_RUN_INTEGRATION") == "1", "Set CLIP_RUN_INTEGRATION=1 for the full ViT-B/32 backward check")
class ViTB32IntegrationTests(unittest.TestCase):
    def test_real_vit_b32_updates_image_text_and_logit_scale(self):
        torch.set_num_threads(2)
        seed_everything(42)
        model, _, _, tokenizer = create_model("ViT-B-32", "cpu")
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        watched = [model.visual.conv1.weight, model.token_embedding.weight, model.logit_scale]
        before = [p.detach().clone() for p in watched]
        output = model(torch.randn(2, 3, 224, 224), tokenizer(["a dog outside", "a red bus on the road"]))
        loss = ClipLoss()(output["image_features"], output["text_features"], output["logit_scale"])
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in watched:
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        optimizer.step()
        for old, parameter in zip(before, watched):
            self.assertFalse(torch.equal(old, parameter))


if __name__ == "__main__":
    unittest.main()
