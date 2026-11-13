# Training and evaluation

This branch uses Transformers' `CLIPModel` and `Trainer` throughout. The
Trainer owns optimization, scheduling, evaluation, saving and resuming; its
built-in `EarlyStoppingCallback` owns stopping. Our code supplies image-caption
batches and retrieval metrics. The custom stopping adapter has been removed.

## Workstation setup

From the repository root:

```bash
uv sync --locked
uv run utilities/download_coco.py
uv run utilities/prepare_data.py
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

For a dataset elsewhere, pass `--data-dir /scratch/coco` to both utility scripts.
Preparation writes `data/clip/{train,validation,test}.csv` with absolute paths,
so rerun it on the workstation. The split has 118,287 training, 1,000 validation
and 4,000 test images. There is no calibration set.

The model starts from random weights in both encoders. Only tokenizer files
are fetched from `openai/clip-vit-base-patch32`. For an offline workstation,
download those files beforehand and pass their local directory via `--tokenizer`.

## Run the three models

```bash
screen -S clip-training
uv run python scripts/train_models.py --epochs 100
```

The suite runs seeds 42, 43 and 44 sequentially, without dropout. Each member
uses batch size 128, CUDA AMP and native early stopping. Training stops after
10 consecutive epochs without improved validation loss, with 100 epochs as
the maximum. A stopped member finishes normally and the next member starts.

For only one member:

```bash
uv run python -m uncertainty_clip.train \
  --epochs 100 --seed 42 --name baseline-42 --resume latest
```

To change the stopping settings, add, for example:

```text
--early-stopping-patience 5 --early-stopping-min-delta 0.001
```

Early stopping monitors `eval_loss`, with lower values better. Minimum delta
is an absolute loss difference. Model selection uses the lowest validation
loss; the minimum delta controls whether an improvement resets patience.
Retrieval recall is reported separately against the entire validation pool.
Each image is paired with its one fixed caption rather than evaluated using
an all-caption COCO benchmark.

Detach with Ctrl+A, then D; reconnect with `screen -r clip-training`. Screen
survives SSH disconnects, not workstation shutdown. After a shutdown, rerun
the command to resume from the most recent checkpoint. CPU checks can use
`--device cpu --precision fp32 --workers 0 --batch-size 2` with a small dataset.
The production recipe supports a single GPU process; distributed contrastive
negative gathering has not been configured.

## Checkpoints and resume

The default output directory is `runs/transformers-suite/baseline-<seed>/`.
Trainer writes native `checkpoint-<step>` directories each epoch, retains the
best and latest, and restores the best model at the end. The entry point then
exports the selected model and tokenizer to `best/`.

The full checkpoints include optimizer, scheduler, RNG, and early-stopping
callback state. `--resume latest` finds the latest checkpoint and restores
patience as well as training progress. Already completed and early-stopped
runs are skipped. Keep the full directory when moving machines and preserve
training settings, dataset order, batch size and the original epoch limit.
A new experiment needs a new run name or `CLIP_LOGS` directory.

The previous trainer's checkpoints are incompatible with this implementation.
Use this branch's new output directory and start fresh. Existing prepared CSVs
have the same format, but regenerate their absolute paths on the workstation.
Framework initialization and optimizer grouping differ from the earlier
implementation; do not treat their training histories as interchangeable.

Load the selected model later using:

```python
from transformers import AutoTokenizer, CLIPModel

path = "runs/transformers-suite/baseline-42/best"
model = CLIPModel.from_pretrained(path).eval()
tokenizer = AutoTokenizer.from_pretrained(path)
```

Image preprocessing is defined in `CaptionDataset`. Only evaluate the test CSV
after the training recipe and validation-based model selection are fixed.

## TensorBoard on your laptop

In a separate terminal or screen session on the workstation:

```bash
screen -S clip-tensorboard
uv run tensorboard --logdir runs/transformers-suite --host 127.0.0.1 --port 6006
```

On your laptop, keep this tunnel running, using your SSH address:

```bash
ssh -N -L 6006:127.0.0.1:6006 username@workstation
```

Open http://localhost:6006. Training loss and learning rate appear during
training; validation loss and recall appear each epoch. The native Trainer
state records the selected checkpoint and stopping callback's patience count.

## Checks

```bash
uv run python -m unittest discover -s tests -v
```

Tiny fixtures exercise real parameter updates in both towers, validation,
retrieval, best-model loading, early stopping and resumed patience. They do
not establish full-data convergence or GPU performance.
