"""Measure real training updates without writing model checkpoints."""

from dataclasses import replace
import gc
import json
import os
from pathlib import Path
import time

from open_clip.loss import ClipLoss
import torch
from torch.utils.data import DataLoader

from .config import SUPPORTED_MODELS
from .data import CocoImages, check_images, file_sha256, load_records, subset, write_json
from .model import create_model, resolve_device, seed_everything, synchronize, use_amp
from .training import environment_metadata, make_optimizer, seed_worker, training_step


def project_training(seconds_per_step, images, batch_size, epochs):
    """Match the runner's dropped final batch, without guessing wall-time overhead."""
    steps_per_epoch = images // batch_size
    if steps_per_epoch < 1:
        raise ValueError("Training split/subset is smaller than batch_size")
    return {
        "images": images,
        "images_per_epoch": steps_per_epoch * batch_size,
        "dropped_images_per_epoch": images % batch_size,
        "epochs": epochs,
        "steps": steps_per_epoch * epochs,
        "training_seconds": seconds_per_step * steps_per_epoch * epochs,
    }


def benchmark_one(config, device, records, warmup_steps, steps):
    seed_everything(config.seed)
    amp = use_amp(config.precision, device)
    model, transform, _, tokenizer = create_model(config.model, device)
    model.train()
    optimizer = make_optimizer(model, config)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    criterion = ClipLoss()
    dataset = CocoImages(config.data_dir, "train", records, transform, config.seed, training=True)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True,
                        num_workers=config.workers, worker_init_fn=seed_worker,
                        generator=torch.Generator().manual_seed(config.seed),
                        pin_memory=device.type == "cuda", persistent_workers=False)
    iterator = iter(loader)

    def update():
        images, captions, image_ids = next(iterator)
        if len(image_ids.unique()) != len(image_ids):
            raise ValueError("Duplicate image IDs in a contrastive batch")
        return training_step(model, tokenizer, images, captions, optimizer, scaler,
                             criterion, device, amp, config.grad_clip_norm)

    warmup_skipped = 0
    for _ in range(warmup_steps):
        _, skipped = update()
        warmup_skipped += skipped
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    losses, skipped_updates = [], 0
    for _ in range(steps):
        loss, skipped = update()
        losses.append(loss)
        skipped_updates += skipped
    synchronize(device)
    elapsed = time.perf_counter() - started
    return {
        "model": config.model,
        "status": "ok" if skipped_updates == 0 else "unstable",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "precision": "cuda_amp_float16" if amp else "fp32",
        "measured_steps": steps,
        "warmup_steps": warmup_steps,
        "warmup_skipped_optimizer_updates": warmup_skipped,
        "skipped_optimizer_updates": skipped_updates,
        "mean_loss": sum(losses) / steps,
        "measured_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "images_per_second": steps * config.batch_size / elapsed,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None,
    }


def benchmark_models(config, models=SUPPORTED_MODELS, warmup_steps=5, steps=30, output=None):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("Benchmark one process/device; do not launch with distributed torchrun")
    if warmup_steps < 1 or steps < 1:
        raise ValueError("Use at least one warm-up step and one measured step")
    if not models or len(set(models)) != len(models) or any(name not in SUPPORTED_MODELS for name in models):
        raise ValueError(f"Choose unique model names from {SUPPORTED_MODELS}")
    if output and Path(output).exists():
        raise ValueError(f"Benchmark output already exists: {output}")
    torch.set_num_threads(config.cpu_threads)
    device = resolve_device(config.device)
    use_amp(config.precision, device)
    config = replace(config, data_dir=str(Path(config.data_dir).resolve()),
                     manifest=str(Path(config.manifest).resolve()))
    records = load_records(config.data_dir, config.manifest)
    train_records = subset(records["train"], config.train_limit, config.seed)
    required = (warmup_steps + steps) * config.batch_size
    if required > len(train_records):
        raise ValueError(f"Benchmark needs {required} unique training images; config selects {len(train_records)}. "
                         "Reduce --steps, --warmup-steps, or --batch-size, or use a larger training config.")
    selected = subset(train_records, required, config.seed)
    check_images(config.data_dir, "train", selected)
    report = {
        "environment": environment_metadata(device),
        "config": config.to_dict(),
        "manifest_sha256": file_sha256(config.manifest),
        "sample_image_ids": [record.image_id for record in selected],
        "estimate_scope": "training only: image loading, augmentation, tokenization, forward/backward, and AdamW",
        "excluded": ["model initialization", "validation", "checkpoint writes", "dataset transfer", "queue time"],
        "notes": [
            "Estimates apply only to the measured device, batch size, precision, workers, and storage.",
            "Short-run data caching and loader prefetch can differ from full-epoch behavior.",
            "Stock RN50 and ViT configurations differ in text width and embedding dimension too.",
            "This measures runtime and memory, not model quality or epochs needed for convergence.",
        ],
        "results": [],
    }
    if device.type != "cuda":
        print(f"Measuring {device}: these estimates do not predict CUDA GPU training time.", flush=True)
    for name in models:
        print(f"Benchmarking {name}: {warmup_steps} warm-up + {steps} measured steps, "
              f"batch size {config.batch_size} on {device}...", flush=True)
        try:
            result = benchmark_one(replace(config, model=name), device, selected, warmup_steps, steps)
        except torch.OutOfMemoryError as error:
            result = {"model": name, "status": "out_of_memory", "error": str(error)}
        except FloatingPointError as error:
            result = {"model": name, "status": "nonfinite", "error": str(error)}
        finally:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if result["status"] == "ok":
            rate = result["seconds_per_step"]
            result["estimates"] = {
                "configured_run": project_training(rate, len(train_records), config.batch_size, config.epochs),
                "full_data_five_epochs": project_training(rate, len(records["train"]), config.batch_size, 5),
            }
            if len(records["train"]) >= 8192:
                result["estimates"]["pilot_three_epochs"] = project_training(rate, 8192, config.batch_size, 3)
            hours = result["estimates"]["full_data_five_epochs"]["training_seconds"] / 3600
            memory = result["peak_cuda_allocated_bytes"]
            memory_label = f"{memory / 2**30:.2f} GiB" if memory is not None else "not measured on CPU/MPS"
            print(f"  {result['images_per_second']:.1f} images/sec; five full-data epochs: "
                  f"{hours:.2f} hours of training; peak allocated GPU memory: {memory_label}", flush=True)
        else:
            print(f"  {result['status']}: no time estimate. "
                  "For memory failures, reduce --batch-size and rerun all models for comparison.", flush=True)
        report["results"].append(result)
        if output:
            write_json(output, report)
    print("Training-time estimates exclude evaluation and checkpoint saving.", flush=True)
    if output:
        print(f"Report saved to {output}", flush=True)
    else:
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return report
