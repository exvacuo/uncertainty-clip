"""Model construction and device setup for the pinned OpenCLIP implementation."""

from contextlib import nullcontext
import random

import numpy as np
import open_clip
import torch

from .config import SUPPORTED_MODELS


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu for a smoke check")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable")
    return torch.device(name)


def use_amp(precision, device):
    if precision == "amp" and device.type != "cuda":
        raise ValueError("AMP requires CUDA; use precision='fp32' on CPU or MPS")
    return device.type == "cuda" and precision in {"auto", "amp"}


def autocast(enabled):
    return torch.autocast("cuda", dtype=torch.float16) if enabled else nullcontext()


def create_model(name, device):
    if name not in SUPPORTED_MODELS:
        raise ValueError(f"Only the built-in {SUPPORTED_MODELS} architectures are supported")
    model, train_transform, eval_transform = open_clip.create_model_and_transforms(
        name,
        pretrained=None,
        load_weights=False,
        pretrained_image=False,
        pretrained_text=False,
        precision="fp32",
        device=device,
        output_dict=True,
    )
    tokenizer = open_clip.get_tokenizer(name)
    return model, train_transform, eval_transform, tokenizer


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
