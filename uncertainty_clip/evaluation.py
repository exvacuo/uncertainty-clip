"""Retrieval with every reference caption treated as a valid image match."""

import hashlib
import json

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import CocoImages
from .model import autocast


def retrieval_metrics(image_features, text_features, image_ids, caption_image_ids, chunk_size=256):
    """Exact Recall@K in [0, 1], bounded-memory scoring, stable candidate-order ties."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    images = image_features.detach().float().cpu()
    texts = text_features.detach().float().cpu()
    image_ids = torch.as_tensor(image_ids, dtype=torch.long).cpu()
    caption_ids = torch.as_tensor(caption_image_ids, dtype=torch.long).cpu()
    if images.ndim != 2 or texts.ndim != 2 or images.shape[1] != texts.shape[1]:
        raise ValueError("Image and text embeddings must have the same feature dimension")
    if len(images) != len(image_ids) or len(texts) != len(caption_ids) or not len(images) or not len(texts):
        raise ValueError("Nonempty features and matching image IDs are required")
    if len(image_ids.unique()) != len(image_ids):
        raise ValueError("Retrieval must encode each image exactly once")
    if set(image_ids.tolist()) != set(caption_ids.tolist()):
        raise ValueError("Every image needs a caption and every caption needs an image")
    for features in (images, texts):
        if not torch.isfinite(features).all() or (features.norm(dim=1) == 0).any():
            raise ValueError("Nonfinite or zero retrieval embeddings")
    images, texts = F.normalize(images, dim=-1), F.normalize(texts, dim=-1)
    result = {"num_images": len(images), "num_captions": len(texts)}
    for prefix, queries, candidates, query_ids, candidate_ids in (
        ("image_to_text", images, texts, image_ids, caption_ids),
        ("text_to_image", texts, images, caption_ids, image_ids),
    ):
        hits = {k: 0 for k in (1, 5, 10)}
        for start in range(0, len(queries), chunk_size):
            scores = queries[start:start + chunk_size] @ candidates.T
            top = scores.argsort(dim=1, descending=True, stable=True)[:, :min(10, len(candidates))]
            matches = candidate_ids[top] == query_ids[start:start + chunk_size, None]
            for k in hits:
                hits[k] += matches[:, :min(k, len(candidates))].any(dim=1).sum().item()
        result.update({f"{prefix}_R@{k}": count / len(queries) for k, count in hits.items()})
    result["mean_recall"] = sum(result[f"{d}_R@{k}"] for d in ("image_to_text", "text_to_image") for k in (1, 5, 10)) / 6
    pool = {"image_ids": image_ids.tolist(), "caption_image_ids": caption_ids.tolist()}
    result["candidate_pool_sha256"] = hashlib.sha256(json.dumps(pool, sort_keys=True).encode()).hexdigest()
    return result


@torch.inference_mode()
def evaluate_retrieval(model, tokenizer, transform, records, data_dir, device, batch_size=128, workers=0, amp=False):
    was_training = model.training
    model.eval()
    try:
        dataset = CocoImages(data_dir, "val", records, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers,
                            pin_memory=device.type == "cuda")
        images, image_ids = [], []
        for batch, ids in loader:
            with autocast(amp):
                features = model.encode_image(batch.to(device, non_blocking=True), normalize=True)
            images.append(features.float().cpu())
            image_ids.extend(ids.tolist())
        captions = [(record.image_id, caption) for record in records for caption in record.captions]
        texts = []
        for start in range(0, len(captions), batch_size):
            tokens = tokenizer([caption for _, caption in captions[start:start + batch_size]]).to(device)
            with autocast(amp):
                features = model.encode_text(tokens, normalize=True)
            texts.append(features.float().cpu())
        return retrieval_metrics(torch.cat(images), torch.cat(texts), image_ids, [i for i, _ in captions])
    finally:
        model.train(was_training)
