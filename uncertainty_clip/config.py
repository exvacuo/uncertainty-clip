"""Experiment configuration; this module needs only the standard library."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path
import tomllib

SUPPORTED_MODELS = ("ViT-B-32", "ViT-B-16", "RN50")


@dataclass(frozen=True)
class TrainConfig:
    data_dir: str = "data/coco"
    manifest: str = "data/splits/coco2017.json"
    output_dir: str = "runs/vit-b32"
    model: str = "ViT-B-32"
    seed: int = 42
    epochs: int = 5
    batch_size: int = 128
    eval_batch_size: int = 128
    workers: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 0.2
    warmup_steps: int = 100
    grad_clip_norm: float = 1.0
    device: str = "auto"
    precision: str = "auto"
    train_limit: int = 0
    validation_limit: int = 0
    log_every: int = 20
    cpu_threads: int = 4

    def __post_init__(self):
        if self.model not in SUPPORTED_MODELS:
            raise ValueError(f"model must be one of {SUPPORTED_MODELS} (random initialization only)")
        for key in ("epochs", "batch_size", "eval_batch_size", "log_every", "cpu_threads"):
            value = getattr(self, key)
            if type(value) is not int or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("batch_size must be at least 2 for contrastive learning")
        for key in ("seed", "workers", "warmup_steps", "train_limit", "validation_limit"):
            value = getattr(self, key)
            if type(value) is not int or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer")
        for key in ("learning_rate", "weight_decay", "grad_clip_norm"):
            value = getattr(self, key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate == 0 or self.grad_clip_norm == 0:
            raise ValueError("learning_rate and grad_clip_norm must be positive")
        if self.precision not in {"auto", "fp32", "amp"}:
            raise ValueError("precision must be auto, fp32, or amp (CUDA float16)")
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto, cpu, cuda, or mps")
        for key in ("data_dir", "manifest", "output_dir"):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be a nonempty path string")

    @classmethod
    def load(cls, path, **overrides):
        with Path(path).open("rb") as stream:
            values = tomllib.load(stream)
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)

    def to_dict(self):
        return asdict(self)


def learning_rate_at(step, total_steps, warmup_steps, base_lr):
    """Linear warmup followed by cosine decay; step is zero based."""
    if total_steps < 1 or not 0 <= warmup_steps < total_steps:
        raise ValueError("warmup_steps must be smaller than the total training steps")
    if not 0 <= step < total_steps:
        raise ValueError("step is outside the training schedule")
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))
