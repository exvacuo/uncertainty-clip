"""Single-device training, epoch-boundary resume, and experiment artifacts."""

from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import open_clip
from open_clip.loss import ClipLoss
import torch
from torch.utils.data import DataLoader

from .config import TrainConfig, learning_rate_at
from .data import check_images, file_sha256, load_records, subset, write_json, CocoImages
from .evaluation import evaluate_retrieval
from .model import autocast, create_model, resolve_device, seed_everything, synchronize, use_amp


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def emit(path, event):
    line = json.dumps(event, sort_keys=True, allow_nan=False)
    with Path(path).open("a") as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def environment_metadata(device):
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {name: version(name) for name in ("torch", "torchvision", "open-clip-torch", "numpy", "pillow")},
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine(),
        "cuda": torch.version.cuda,
        "git_revision": revision,
        "git_dirty": dirty,
    }


def make_optimizer(model, config):
    no_decay, decay = [], []
    excluded = model.no_weight_decay()
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name in excluded else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": config.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=config.learning_rate, betas=(0.9, 0.98), eps=1e-6,
    )


def training_step(model, tokenizer, images, captions, optimizer, scaler, criterion, device, amp, grad_clip_norm):
    """The same update is used for training and hardware timing."""
    optimizer.zero_grad(set_to_none=True)
    images = images.to(device, non_blocking=True)
    tokens = tokenizer(list(captions)).to(device)
    with autocast(amp):
        prediction = model(images, tokens)
        loss = criterion(prediction["image_features"], prediction["text_features"], prediction["logit_scale"])
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
    if not amp and not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite training gradients")
    previous_scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    skipped = int(scaler.get_scale() < previous_scale)
    with torch.no_grad():
        model.logit_scale.clamp_(0, math.log(100))
    return loss.item(), skipped


def train(config, resume=False):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pilot runner supports one process/device; do not launch with distributed torchrun")
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    amp = use_amp(config.precision, device)
    # Store absolute paths for unambiguous resume and checkpoint evaluation.
    config = replace(config, **{key: str(Path(getattr(config, key)).resolve())
                               for key in ("data_dir", "manifest", "output_dir")})
    output = Path(config.output_dir)
    if resume:
        if not (output / "last.pt").is_file():
            raise FileNotFoundError(f"No last.pt checkpoint in {output}")
    elif output.exists() and any(output.iterdir()):
        raise ValueError(f"Run directory is not empty: {output}; choose a new output or use --resume")
    records = load_records(config.data_dir, config.manifest)
    train_records = subset(records["train"], config.train_limit, config.seed)
    validation_records = subset(records["validation"], config.validation_limit, config.seed)
    steps_per_epoch = len(train_records) // config.batch_size
    if steps_per_epoch < 1:
        raise ValueError("Training split/subset is smaller than batch_size")
    total_steps = steps_per_epoch * config.epochs
    learning_rate_at(0, total_steps, config.warmup_steps, config.learning_rate)
    for source, selected in (("train", train_records), ("val", validation_records)):
        check_images(config.data_dir, source, selected)
    manifest_hash = file_sha256(config.manifest)
    seed_everything(config.seed)
    model, train_transform, eval_transform, tokenizer = create_model(config.model, device)
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = ClipLoss()
    start_epoch, global_step, best_score = 0, 0, -1.0
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
        if checkpoint["config"] != config.to_dict() or checkpoint["manifest_sha256"] != manifest_hash:
            raise ValueError("Resume requires the same configuration and split manifest")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, global_step, best_score = checkpoint["epoch"], checkpoint["global_step"], checkpoint["best_score"]
        del checkpoint
    if start_epoch >= config.epochs:
        print("Run already completed all configured epochs.", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "metrics.jsonl"
    metadata = {
        **environment_metadata(device),
        "config": config.to_dict(),
        "manifest_sha256": manifest_hash,
        "initialization": {"image": "random", "text": "random", "pretrained": None},
        "tokenizer": type(tokenizer).__name__,
        "context_length": model.context_length,
        "model_config": open_clip.get_model_config(config.model),
        "source_sha256": {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))},
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "train_transform": repr(train_transform),
        "eval_transform": repr(eval_transform),
        "precision_resolved": "cuda_amp_float16" if amp else "fp32",
        "train_images": len(train_records),
        "validation_images": len(validation_records),
        "train_images_per_epoch": steps_per_epoch * config.batch_size,
        "dropped_images_per_epoch": len(train_records) % config.batch_size,
        "checkpoint_selection": "maximum validation mean Recall@1/5/10 across both retrieval directions; ties keep earlier epoch",
    }
    if not resume:
        write_json(output / "config.json", config.to_dict())
        write_json(output / "metadata.json", metadata)
        write_json(output / "splits.json", json.loads(Path(config.manifest).read_text()))
        write_json(output / "selected-images.json", {
            "train": [r.image_id for r in train_records], "validation": [r.image_id for r in validation_records]
        })
        initial = evaluate_retrieval(model, tokenizer, eval_transform, validation_records, config.data_dir,
                                     device, config.eval_batch_size, config.workers, amp)
        emit(log_path, {"event": "initial_validation", "epoch": 0, **initial})
    else:
        emit(log_path, {"event": "resume", "epoch": start_epoch, "global_step": global_step,
                        "environment": environment_metadata(device)})
    dataset = CocoImages(config.data_dir, "train", train_records, train_transform, config.seed, training=True)
    for epoch in range(start_epoch, config.epochs):
        # Epoch-local RNGs make epoch-boundary resume reproducible on the same stack.
        seed_everything(config.seed + epoch)
        dataset.epoch = epoch
        loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                            num_workers=config.workers, worker_init_fn=seed_worker,
                            generator=torch.Generator().manual_seed(config.seed + epoch),
                            pin_memory=device.type == "cuda", persistent_workers=False)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        total_loss, seen, skipped_updates = 0.0, 0, 0
        for batch_index, (images, captions, image_ids) in enumerate(loader):
            if len(image_ids.unique()) != len(image_ids):
                raise ValueError("Duplicate image IDs in a contrastive batch")
            lr = learning_rate_at(global_step, total_steps, config.warmup_steps, config.learning_rate)
            for group in optimizer.param_groups:
                group["lr"] = lr
            value, skipped = training_step(model, tokenizer, images, captions, optimizer, scaler,
                                          criterion, device, amp, config.grad_clip_norm)
            skipped_updates += skipped
            count = len(images)
            total_loss += value * count
            seen += count
            global_step += 1
            if (batch_index + 1) % config.log_every == 0 or batch_index + 1 == steps_per_epoch:
                emit(log_path, {"event": "train_step", "epoch": epoch + 1, "global_step": global_step,
                                "loss": value, "learning_rate": lr, "logit_scale": model.logit_scale.exp().item()})
        synchronize(device)
        train_seconds = time.perf_counter() - started
        peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        validation = evaluate_retrieval(model, tokenizer, eval_transform, validation_records, config.data_dir,
                                        device, config.eval_batch_size, config.workers, amp)
        event = {"event": "epoch", "epoch": epoch + 1, "global_step": global_step,
                 "train_loss": total_loss / seen, "train_images": seen, "train_seconds": train_seconds,
                 "train_images_per_second": seen / train_seconds, "peak_cuda_memory_bytes": peak_memory,
                 "skipped_optimizer_updates": skipped_updates, "validation": validation}
        is_best = validation["mean_recall"] > best_score
        best_score = max(best_score, validation["mean_recall"])
        payload = {"format_version": 1, "config": config.to_dict(), "manifest_sha256": manifest_hash,
                   "model": model.state_dict(), "epoch": epoch + 1, "global_step": global_step,
                   "best_score": best_score, "validation": validation}
        if is_best:
            save_checkpoint(output / "best.pt", payload)
        save_checkpoint(output / "last.pt", {**payload, "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()})
        emit(log_path, event)
        write_json(output / "summary.json", event | {"best_mean_recall": best_score})


def evaluate_checkpoint(checkpoint_path, split="validation", data_dir=None, manifest=None, device_name="auto",
                        batch_size=128, workers=0, limit=0, output=None):
    if split not in {"validation", "calibration", "test"}:
        raise ValueError("Evaluate validation, calibration, or test")
    if batch_size < 1 or workers < 0 or limit < 0:
        raise ValueError("Use a positive batch size and nonnegative workers/limit")
    if output and Path(output).exists():
        raise ValueError(f"Evaluation output already exists: {output}")
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = TrainConfig(**checkpoint["config"])
    torch.set_num_threads(config.cpu_threads)
    data_dir = data_dir or config.data_dir
    manifest = manifest or config.manifest
    if file_sha256(manifest) != checkpoint["manifest_sha256"]:
        raise ValueError("Evaluation split manifest differs from training")
    records = subset(load_records(data_dir, manifest)[split], limit, config.seed)
    check_images(data_dir, "val", records)
    device = resolve_device(device_name)
    seed_everything(config.seed)
    model, _, transform, tokenizer = create_model(config.model, device)
    model.load_state_dict(checkpoint["model"])
    epoch = checkpoint["epoch"]
    del checkpoint
    metrics = evaluate_retrieval(model, tokenizer, transform, records, data_dir, device, batch_size, workers)
    result = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": file_sha256(checkpoint_path),
              "epoch": epoch, "split": split, "manifest_sha256": file_sha256(manifest),
              "precision": "fp32", "environment": environment_metadata(device), **metrics}
    if output:
        write_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return result
