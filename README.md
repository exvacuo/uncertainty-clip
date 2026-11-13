# Uncertainty Quantification with CLIP

Train three CLIP ViT-B/32 models from random weights on COCO 2017, with seeds
42, 43 and 44 and no dropout. The models form a deep ensemble for later
uncertainty experiments. MC dropout is outside the current training scope.

The implementation uses Hugging Face Transformers' `CLIPModel`, contrastive
loss, `Trainer`, and native `EarlyStoppingCallback`. There is no custom
training loop, optimizer implementation, stopping callback, or runtime patching.
Project code prepares image-caption pairs and computes retrieval metrics.

## Setup and data

From the repository root:

```bash
uv sync --locked
uv run utilities/download_coco.py
uv run utilities/prepare_data.py
```

Preparation writes `data/clip/train.csv`, `validation.csv`, and `test.csv`.
Paths are absolute: regenerate the CSVs after moving the dataset. For a custom
location, use `--data-dir /scratch/coco` with both utility scripts.

| Split | Images | Purpose |
|---|---:|---|
| Training | 118,287 | Model optimization |
| Validation | 1,000 | Early stopping and checkpoint selection |
| Test | 4,000 | Final evaluation only |

Validation and test divide COCO val2017 using seed 42, with no shared images.
There is no calibration set. Each image uses one fixed caption. This is a
custom COCO 2017 split, not the Karpathy retrieval benchmark.

## Training

All three models, sequentially:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/train_models.py
```

One model, for up to 100 epochs:

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m uncertainty_clip.train \
  --seed 42 --name baseline-42 --epochs 100 --resume latest
```

Both commands enable TensorBoard and native early stopping by default. Each
model stops after 10 consecutive validation evaluations without improvement
in validation loss. Configure this with `--early-stopping-patience` and
`--early-stopping-min-delta`. The three-model command forwards options to the
same training entry point; leave seed and run name to the launcher.

| Setting | Value |
|---|---|
| Architecture | ViT-B/32: 12 layers in each encoder, 512-dimensional embeddings |
| Initialization | Random weights in both encoders; all parameters trainable |
| Dropout | Zero in both encoders |
| Images | 224 × 224; random resized crop for training, resize/center crop for validation |
| Text | CLIP BPE tokenizer, maximum 77 tokens |
| Batch size | 128 |
| Optimizer | Trainer's PyTorch AdamW, betas (0.9, 0.98), epsilon 1e-6 |
| Learning rate / weight decay | 5e-4 / 0.2 |
| Schedule | 20 optimizer warmup steps, then cosine decay |
| Gradient clipping | 1.0 |
| Precision | CUDA AMP |
| Epoch limit | 100 |
| Early stopping | Validation loss; patience 10, minimum improvement 0 |

`CLIPModel` is constructed from configuration, never pretrained model weights.
The first run downloads tokenizer files only; cache them before an offline run.
Unlike the earlier trainer, this recipe includes the final partial training
batch. The framework's initialization and optimizer grouping also differ, so
this is a new experiment, not a numerically identical continuation.

Set `CUDA_VISIBLE_DEVICES` to one GPU assigned to you (replace `0` as needed).
Trainer otherwise enables DataParallel when multiple GPUs are visible; its
output gathering is incompatible with CLIP's unequal pairwise-logit matrices.
The entry point rejects this configuration before loading the model.

The recipe supports one training process on one GPU, with CPU options for
pipeline checks. Full GPU memory use, throughput and convergence still need
validation. The old runtime estimates do not establish this trainer's speed.

## Checkpoints and progress

Outputs live under `runs/transformers-suite/baseline-<seed>/`:

- `checkpoint-<step>/`: native model, optimizer, scheduler, RNG and callback state.
- `best/`: model and tokenizer exported after training loads the best checkpoint.
- `runs/`: TensorBoard training and validation events in timestamped directories.
- `trainer_state.json`: best-checkpoint location and metric history.

Trainer retains the best and latest checkpoints (`save_total_limit=2`), rather
than keeping every epoch. Resume with `--resume latest`; a completed or
early-stopped member is skipped when the suite is repeated. Keep the complete
run directory and unchanged training settings when resuming. Use a new name or
`CLIP_LOGS` directory for a different experiment. Checkpoints from the previous
training implementation cannot be resumed by this Trainer.

An intentional continuation after stopping can use `--reset-early-stopping`
with `--resume` and a larger patience. This resets the native callback while
preserving model and optimizer/scheduler progress. See the
[continuation example](docs/training.md); keep the original run and use a
separate output directory for the revised experiment.

Validation logs include loss and image-to-text/text-to-image Recall@1, @5 and
@10. Recall uses the entire validation candidate pool; loss uses in-batch
negatives, so keep the evaluation batch size fixed when comparing loss.
The test CSV is never read during training or checkpoint selection.

```bash
uv run tensorboard --logdir runs/transformers-suite --host 127.0.0.1 --port 6006
```

See [the training guide](docs/training.md) for persistent sessions and access
from a laptop.

## Verification

```bash
uv run python -m unittest discover -s tests -v
```

Tests use tiny local models and a fixture tokenizer without Hub downloads.
They exercise real Trainer updates, validation over incomplete batches,
retrieval, native early stopping, saved best models and resumed callback state.

Sources: [Transformers CLIP](https://huggingface.co/docs/transformers/en/model_doc/clip),
[Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer),
[EarlyStoppingCallback](https://huggingface.co/docs/transformers/en/trainer_callbacks).
