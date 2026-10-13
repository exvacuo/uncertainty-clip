# CLIP baseline: run guide

## Implementation and current validation status

The baseline wraps **OpenCLIP 3.2.0's ViT-B/32**, including its vision Transformer,
CLIP text Transformer, bundled BPE tokenizer, preprocessing, and symmetric
`ClipLoss`. PyTorch is pinned to 2.8.0 and torchvision to 0.23.0. Model creation
explicitly sets `pretrained=None`, `load_weights=False`, `pretrained_image=False`,
and `pretrained_text=False`. Both encoders and the learned logit scale train
together. No model weights or tokenizer models need downloading at runtime for
these built-in architectures once the Python packages are installed.

Implementation references:

- [Pinned OpenCLIP release](https://github.com/mlfoundations/open_clip/releases/tag/v3.2.0)
- [Model and transform factory](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip/factory.py)
- [Symmetric contrastive loss](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip/loss.py)
- [ViT-B/32 configuration](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip/model_configs/ViT-B-32.json)

The data preparation has run against the local COCO dataset. Standard-library
tests cover split isolation, annotation fingerprints, missing images,
deterministic caption sampling, configuration, and the learning-rate schedule.
Dependencies are installed and the lockfile is current. On 2026-10-13, all
22 tests passed with `CLIP_RUN_INTEGRATION=1`, with no skips, including a real
ViT-B/32 CPU backward update, checkpoint/resume tests, and benchmark failure
handling. The four-image real-data CPU check saved checkpoints and retrieval
metrics under `runs/vit-b32-smoke/`. All three models also completed the short
CPU benchmark. CUDA timing and the GPU pilot remain pending.
There are **no pilot or full-data training results yet**. Do not interpret
the configurations as tuned recipes.

## Environment

Run commands from the repository root. The selected Python version is 3.13.

```bash
uv sync
```

The lockfile includes the training dependencies. Use `uv sync --locked` to
reproduce the environment. On an NVIDIA machine, confirm that the installed
PyTorch build can use its GPU:

```bash
uv run python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

`device="auto"` selects CUDA when available, otherwise CPU. Apple MPS is
available only by explicit `--device mps`; its model compatibility is untested.
`precision="auto"` uses CUDA float16 autocast with gradient scaling on CUDA
and float32 elsewhere. Explicit `precision="amp"` requires CUDA.

## Dataset and split protocol

```bash
uv run utilities/download_coco.py
uv run python -m uncertainty_clip prepare
```

The default manifest is `data/splits/coco2017.json`, using seed 42:

| Purpose | Source | Images |
| --- | --- | ---: |
| Training | COCO train2017 | 118,287 |
| Model selection (`validation`) | COCO val2017 | 1,000 |
| Calibration | COCO val2017 | 1,000 |
| Final test | COCO val2017 | 3,000 |

The held-out image IDs are shuffled with a local seeded RNG before partitioning.
Every caption stays with its image. The manifest records annotation SHA-256
hashes. Repeating preparation is idempotent; changing the split seed or sizes
requires a different `--output` path. Training and evaluation reject duplicate
IDs, overlapping splits, or changed annotations. Preparation checks that each
image file exists; decoding during loading will detect corrupt images.

This is a **custom COCO 2017 protocol**, not the Karpathy split or a published
COCO retrieval benchmark. Do not compare its Recall@K directly to papers using
different images or candidate pools. Training only evaluates `validation`.
Reserve `calibration` for fitting uncertainty parameters and `test` for the
final evaluation after selecting the recipe.

One caption is sampled per image per epoch, deterministically from the seed,
epoch, and image ID. The loader shuffles images without replacement. It drops
the incomplete final batch, with counts recorded in metadata. The runner is
single-process and single-device, so IDs remain unique across each entire
contrastive batch. There is no gradient accumulation or distributed training.

For datasets mounted elsewhere:

```bash
uv run python -m uncertainty_clip prepare \
  --data-dir /scratch/datasets/coco --output data/splits/coco2017.json
uv run python -m uncertainty_clip train --config configs/vit-b32-pilot.toml \
  --data-dir /scratch/datasets/coco
```

All relative configuration paths resolve from the working directory.

## Validate the implementation

```bash
# Data checks plus tensor, loss, checkpoint, and resume tests.
uv run python -m unittest discover -s tests -v

# Include an actual ViT-B/32 forward/backward update on CPU.
CLIP_RUN_INTEGRATION=1 uv run python -m unittest discover -s tests -v

# End-to-end run on four real training and four validation images.
uv run python -m uncertainty_clip train --config configs/vit-b32-smoke.toml
uv run python -m uncertainty_clip evaluate \
  --checkpoint runs/vit-b32-smoke/last.pt --device cpu --limit 4 \
  --output runs/vit-b32-smoke/validation.json
```

The integration test checks that ViT-B/32 image weights, text embeddings, and logit
scale actually change. A separate small dual-encoder test exercises the real
runner: it interrupts after an epoch checkpoint, resumes, and compares the final
weights and retrieval metrics with uninterrupted training. Tests requiring
uninstalled packages report skips, so an `OK` with skips is not full validation.

The smoke configuration is a quick pipeline check, not a quality experiment. Recall@5 and Recall@10 become
trivial in its tiny image candidate pool. Use a new `--output` directory to
repeat a completed smoke run; existing run directories are protected against
accidental overwrite.

## Run the pilot and full-data baseline

```bash
uv run python -m uncertainty_clip train --config configs/vit-b32-pilot.toml
uv run python -m uncertainty_clip train --config configs/vit-b32.toml --device cuda
```

The pilot uses a deterministic 8,192-image training subset for three epochs,
batch size 128, and all 1,000 model-selection images. The full-data configuration
uses five epochs over the official training split. These are starting points;
the pilot should establish a feasible batch size and whether held-out retrieval
improves over the initial random model. If a batch does not fit, change the TOML
and use a new output directory. Changing batch size changes the negative pool.

Training uses AdamW with learning rate 5e-4, betas (0.9, 0.98), epsilon 1e-6,
weight decay 0.2, linear warmup followed by cosine decay, and gradient norm
clipping at 1.0. Biases, one-dimensional parameters, and OpenCLIP's declared
exclusions receive no weight decay. The logit scale is clamped to [1, 100] after
each update. Random training crops and deterministic evaluation preprocessing
come from the pinned OpenCLIP model; their full descriptions are saved.

The runner evaluates the random model before training, then evaluates after
each epoch. `best.pt` selects the **completed training epoch** with maximum mean
Recall@1/5/10 over both retrieval directions. Ties retain the earlier epoch.
The initial random model is a diagnostic and is not a selected checkpoint.

Each run creates:

| Artifact | Contents |
| --- | --- |
| `config.json` | Full resolved settings, including absolute paths |
| `metadata.json` | Versions, device, source hashes, model, tokenizer, preprocessing, seed and selection rule |
| `splits.json` | Copy of the full split manifest |
| `selected-images.json` | Exact training and validation subsets |
| `metrics.jsonl` | Initial retrieval, step losses, epoch retrieval and resume events |
| `last.pt` | Model, optimizer, gradient scaler, epoch and schedule position |
| `best.pt` | Selected model checkpoint and its validation scores |
| `summary.json` | Most recent completed epoch and best selection score |

Checkpoints are replaced atomically. Allow several GB for ViT-B/32 checkpoints and
their temporary replacements. `runs/` and generated `data/` files are ignored
by Git. Run artifacts record source hashes and the Git revision for reproducibility.

Training throughput includes data loading, tokenization, and step logging;
it excludes retrieval evaluation and checkpoint writing. CUDA peak allocated
memory is measured for training each epoch. CPU and MPS runs report null for
that CUDA-specific field. Logs also count optimizer updates skipped by the
mixed-precision scaler.

Resume at the last completed epoch with the same configuration:

```bash
uv run python -m uncertainty_clip train --config configs/vit-b32-pilot.toml --resume
```

Work within an interrupted epoch is replayed. Epoch-local RNG seeds reproduce
sampling, ordering, and augmentation on the same software/hardware stack.
Changing epochs, batch size, paths, or other settings during resume is rejected
to preserve the original learning-rate schedule and provenance. A completed run
is a no-op on resume. Bitwise equivalence across different GPU architectures or
library versions is not promised.

## GPU handoff and compute estimates

The successful CPU check makes the project ready for a GPU pilot. Transfer the
source, `pyproject.toml`, refreshed `uv.lock`, configurations, COCO data, and
`data/splits/coco2017.json`. On the GPU machine, run `uv sync --locked`, check
`torch.cuda.is_available()`, then launch:

```bash
time uv run python -m uncertainty_clip train --config configs/vit-b32-pilot.toml
```

If COCO is mounted elsewhere, add `--data-dir /path/to/coco`. Start from the
repository root. The pilot uses one CUDA GPU, mixed precision, batch size 128,
8,192 training images for three epochs, and 1,000 validation images. GPU memory
fit is unmeasured; if batch size needs changing, update the config and use a
fresh output directory.

### Compare compute cost before training

Run a short hardware benchmark on the same GPU and storage intended for training:

```bash
uv run python -m uncertainty_clip benchmark --output runs/compute-estimate.json
```

By default, it compares **ViT-B/32, ViT-B/16, and RN50** sequentially, with
batch size 128, five warm-up steps, and 30 measured updates per model. It uses
real COCO images/captions, the training transforms, and the same AdamW update
code as the training runner. Warm-up initializes kernels and optimizer state.
No trained checkpoints are saved.

The report includes images/second, average step time, peak CUDA allocated and
reserved memory, software/hardware details, data hashes, sampled image IDs,
and estimates for the configured run, the three-epoch pilot, and five full-data
epochs. Failed or unstable models receive no time estimate; a memory failure is
reported and the remaining models are attempted. The command exits nonzero if
any model fails. Batch size is never silently reduced.

For a different dataset mount or smaller batch size:

```bash
uv run python -m uncertainty_clip benchmark \
  --data-dir /scratch/datasets/coco --batch-size 64 \
  --output runs/compute-estimate-b64.json
```

Change `--config`, `--epochs`, `--workers`, `--steps`, `--warmup-steps`, or
`--models ViT-B-16` as needed. Choose a fresh report path each time. Each model
uses identical selected image IDs and batch size. Stock RN50 differs from the
ViTs in text width and embedding dimension too, so this compares whole models.

To check the benchmark command locally without a GPU:

```bash
uv run python -m uncertainty_clip benchmark --device cpu --batch-size 2 \
  --workers 0 --warmup-steps 1 --steps 2 --output runs/compute-estimate-cpu.json
```

CPU measurements predict CPU training only. The default command requires CUDA
and fails clearly if no NVIDIA GPU is available.

Training time is measured seconds per step multiplied by
`floor(training_images / batch_size) × epochs`. The default pilot has 192
updates; the full-data five-epoch configuration has 4,620 updates. Incomplete
final batches are dropped exactly as in the training runner.

**These are training-only estimates.** They include loading, augmentation,
tokenization, forward/backward computation, and optimizer updates. Add model
initialization, retrieval evaluation, and checkpoint writes for total process
time; environment setup, dataset transfer, and queue time are separate. The
benchmark does not time step logging, and short-run caching and loader prefetch
can differ from full-epoch behavior. Confirm estimates with the real pilot.
Five epochs is a starting recipe, not evidence of convergence.

## Retrieval and the final handoff

```bash
uv run python -m uncertainty_clip evaluate \
  --checkpoint runs/vit-b32-pilot/best.pt --split validation --device cuda \
  --output runs/vit-b32-pilot/validation.json

# Run only once the recipe and uncertainty protocol are fixed.
uv run python -m uncertainty_clip evaluate \
  --checkpoint runs/vit-b32/best.pt --split test --device cuda \
  --output runs/vit-b32/test.json
```

Evaluation encodes each image once and every caption. Similarity uses normalized
embeddings. For image-to-text retrieval, any reference caption counts as a hit;
for text-to-image, the caption's image is the target. Recall values are fractions
in [0, 1]. Scoring is chunked, with deterministic candidate-order tie handling.
Results identify the image and caption pool sizes, pool fingerprint, split and
checkpoint hashes. Standalone evaluation uses float32; it may differ slightly
from in-training CUDA mixed-precision retrieval. Compare the same precision and
candidate pool when interpreting small changes.

Use `--data-dir` and `--manifest` when moving a checkpoint to another machine.
The manifest must have the same hash; the run's `splits.json` is a portable copy.
An explicit `--limit` changes the candidate pool and is intended for smoke
checks. Standalone evaluation otherwise uses the entire requested split.

For Matias, the handoff should include the source, refreshed dependency lock,
exact configuration, split manifest, a successful smoke check, and pilot metrics
showing initial versus trained retrieval, throughput, and memory use. Only then
choose whether to train longer, adjust model size, or compare RN50. Stock
RN50 and ViT-B/32 compare whole configurations, including different embedding
dimensions; this is not an isolated image-backbone comparison.
