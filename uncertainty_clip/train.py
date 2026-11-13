"""Train randomly initialized CLIP models with Hugging Face Trainer."""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import (
    AutoTokenizer, CLIPConfig, CLIPModel, EarlyStoppingCallback,
    Trainer, TrainingArguments, set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


class CaptionDataset(Dataset):
    def __init__(self, path, tokenizer, config, training):
        with Path(path).open(newline="", encoding="utf-8") as stream:
            self.rows = list(csv.DictReader(stream))
        self.tokenizer = tokenizer
        self.length = config.text_config.max_position_embeddings
        size = config.vision_config.image_size
        resize = [transforms.RandomResizedCrop(size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC)] \
            if training else [transforms.Resize(size, interpolation=InterpolationMode.BICUBIC), transforms.CenterCrop(size)]
        self.transform = transforms.Compose([
            *resize, transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        tokens = self.tokenizer(row["title"], padding="max_length", truncation=True,
                                max_length=self.length, return_tensors="pt")
        with Image.open(row["filepath"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return {"input_ids": tokens["input_ids"][0], "attention_mask": tokens["attention_mask"][0],
                "pixel_values": pixels}


def collate(samples):
    return {**{key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}, "return_loss": True}


def retrieval_metrics(prediction):
    # Score against the whole held-out pool, not separately within each batch.
    text, image = prediction.predictions
    scores = image @ text.T
    metrics = {}
    for direction, similarities in (("image_to_text", scores), ("text_to_image", scores.T)):
        ranks = np.argsort(-similarities, axis=1)
        for k in (1, 5, 10):
            metrics[f"{direction}_R@{k}"] = float((ranks[:, :k] == np.arange(len(ranks))[:, None]).any(axis=1).mean())
    return metrics


def parse_args(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default=os.environ.get("CLIP_TRAIN_DATA", "data/clip/train.csv"))
    parser.add_argument("--val-data", default=os.environ.get("CLIP_VAL_DATA", "data/clip/validation.csv"))
    parser.add_argument("--model-config", type=Path, help="Local CLIP config JSON; defaults to ViT-B/32")
    parser.add_argument("--tokenizer", default="openai/clip-vit-base-patch32")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--wd", type=float, default=0.2)
    parser.add_argument("--warmup", type=int, default=20, help="Optimizer steps")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("amp", "fp32"), default="amp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--logs", default=os.environ.get("CLIP_LOGS", "runs/transformers-suite"))
    parser.add_argument("--name", default="baseline-42")
    parser.add_argument("--resume", help="latest, or a Transformers checkpoint directory")
    parser.add_argument("--report-to", choices=("tensorboard", "none"), default="tensorboard")
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    args = parser.parse_args(arguments)
    if args.epochs < 1 or args.batch_size < 2 or args.early_stopping_patience < 1:
        parser.error("epochs and patience must be positive; batch size must be at least 2")
    if not np.isfinite(args.early_stopping_min_delta) or args.early_stopping_min_delta < 0:
        parser.error("minimum improvement must be finite and nonnegative")
    return args


def build_trainer(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. For a CPU check, use --device cpu --precision fp32.")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise ValueError("This recipe supports one training process; distributed contrastive negatives are not configured.")
    set_seed(args.seed)
    config = CLIPConfig.from_json_file(args.model_config) if args.model_config else CLIPConfig(
        text_config={"hidden_act": "gelu", "attention_dropout": 0.0},
        vision_config={"hidden_act": "gelu", "patch_size": 32, "attention_dropout": 0.0},
    )
    if config.text_config.attention_dropout or config.vision_config.attention_dropout:
        raise ValueError("This ensemble recipe requires dropout to be zero in both encoders.")
    config.keys_to_ignore_at_inference = ["logits_per_image", "logits_per_text", "text_model_output", "vision_model_output"]
    model = CLIPModel(config)  # Both towers start with random weights.
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    train_data = CaptionDataset(args.train_data, tokenizer, config, training=True)
    val_data = CaptionDataset(args.val_data, tokenizer, config, training=False)
    if len(train_data) < args.batch_size or len(val_data) < 2:
        raise ValueError("Need at least one full training batch and two validation pairs.")
    output = Path(args.logs) / args.name
    training_args = TrainingArguments(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr, weight_decay=args.wd, warmup_steps=args.warmup,
        lr_scheduler_type="cosine", optim="adamw_torch", adam_beta1=0.9, adam_beta2=0.98, adam_epsilon=1e-6,
        max_grad_norm=1.0, fp16=args.precision == "amp" and args.device == "cuda", use_cpu=args.device == "cpu",
        seed=args.seed, data_seed=args.seed, dataloader_num_workers=args.workers,
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        restore_callback_states_from_checkpoint=True, logging_steps=10, logging_first_step=True,
        report_to=args.report_to,
        label_names=["input_ids"], remove_unused_columns=False,
    )
    return Trainer(
        model=model, args=training_args, train_dataset=train_data, eval_dataset=val_data,
        data_collator=collate, processing_class=tokenizer, compute_metrics=retrieval_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience,
                                         early_stopping_threshold=args.early_stopping_min_delta)],
    )


def main(arguments=None):
    args = parse_args(arguments)
    output = Path(args.logs) / args.name
    checkpoint = get_last_checkpoint(str(output)) if output.is_dir() else None
    if args.resume != "latest":
        if checkpoint and not args.resume:
            raise ValueError("Run already has checkpoints. Use --resume latest or choose a new --name.")
        checkpoint = args.resume
    if checkpoint:
        state = json.loads((Path(checkpoint) / "trainer_state.json").read_text())
        callback = state.get("stateful_callbacks", {}).get("EarlyStoppingCallback", {})
        if state.get("epoch", 0) >= args.epochs or callback.get("attributes", {}).get("early_stopping_patience_counter", 0) >= args.early_stopping_patience:
            print(f"{args.name}: training already finished; best checkpoint: {state.get('best_model_checkpoint')}")
            return
    trainer = build_trainer(args)
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(str(output / "best"))
    trainer.save_state()
    trainer.log_metrics("train", {"best_validation_loss": trainer.state.best_metric})


if __name__ == "__main__":
    main()
