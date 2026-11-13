# Training and evaluation

For the three ensemble members without dropout, run from the repository root:

```bash
uv sync --locked
uv run utilities/prepare_data.py
uv run python scripts/train_models.py --epochs 100
```

The suite calls `uncertainty_clip.train` for each member; that wrapper supplies
the shared settings to OpenCLIP's official trainer. It runs ViT-B/32
sequentially with seeds 42, 43 and 44, using CUDA AMP
and batch size 128. It does not launch a dropout model. Each run starts from
random weights unless its latest checkpoint already exists under
`runs/openclip-suite/baseline-<seed>/checkpoints/`. Repeating the command resumes
each member toward 100 total epochs. Use a new `CLIP_LOGS` directory when
changing training settings. Additional CLI flags are passed to OpenCLIP;
keep the model, seed, run name and dropout settings unchanged for this ensemble.

Download COCO first if needed (`uv run utilities/download_coco.py`). Regenerate
the CSVs on the workstation because their image paths are absolute. Check CUDA
before starting:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.cuda.get_device_name(0))"
```

Warmup is 20 optimizer steps, not epochs. The 100-epoch schedule and other
hyperparameters are starting settings; full GPU performance and convergence
still need validation.

For individual runs, preparation and training use:

- `utilities/prepare_data.py` prepares COCO as CSV files before training.
- `uncertainty_clip/train.py` passes those files to OpenCLIP's official trainer.

```bash
uv run utilities/prepare_data.py
uv run python -m uncertainty_clip.train
```

Preparation uses every COCO training image. It reserves 1,000 validation
and 4,000 test images from COCO val2017, using seed 42.
It writes one fixed caption per image. CSV paths are absolute, so re-run
preparation after moving the dataset. For another location:

```bash
uv run utilities/prepare_data.py --data-dir /scratch/coco --output data/openclip
```

OpenCLIP supplies the model, image preprocessing, tokenizer, loss, optimizer,
scheduler, training loop, validation and checkpoint saving. The default flags
in `train.py` request ViT-B/32, three epochs, batch size 128, CUDA and AMP.
No pretrained weights are requested. GPU performance and model quality still
need checking; these are starting settings.

Use a new run name for a new experiment. Resume without changing settings:

```bash
uv run python -m uncertainty_clip.train --resume latest
```

The official trainer evaluates validation each epoch and writes recall metrics
and loss to `runs/vit-b32/checkpoints/results.jsonl`. It saves `epoch_1.pt`, etc.,
and `epoch_latest.pt`. Select an epoch using validation before final test evaluation.
There is no custom evaluation or model implementation in this project.

To evaluate using OpenCLIP directly:

```bash
uv run python -m open_clip_train.main \
  --model ViT-B-32 --pretrained runs/vit-b32/checkpoints/epoch_3.pt \
  --val-data data/openclip/validation.csv --dataset-type csv --csv-separator ','
```

For RN50 or ViT-B/16, use the matching model flag. Evaluation uses the single
caption paired with each image in the CSV, rather than an all-caption COCO
benchmark. Our held-out split is a custom COCO 2017 split, not Karpathy's split.

For a short CPU training check without using the full dataset:

```bash
OMP_NUM_THREADS=2 uv run python -m uncertainty_clip.train \
  --dataset-type synthetic --train-num-samples 4 --val-data '' \
  --device cpu --precision fp32 --batch-size 2 --epochs 1 \
  --workers 0 --warmup 0 --name openclip-smoke
```

Tests check preparation, actual OpenCLIP updates, validation and resume on a tiny
fixture. The full GPU run remains to be done.

```bash
uv run python -m unittest discover -s tests -v
```

Source: [OpenCLIP's official training and evaluation entry point](https://github.com/mlfoundations/open_clip/blob/v3.2.0/src/open_clip_train/main.py).
