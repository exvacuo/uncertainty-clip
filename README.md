# uncertainty-clip

Use OpenCLIP 3.2.0 for the model, training and evaluation.

Run from the repository root:

```bash
uv sync --locked
uv run utilities/prepare_data.py
uv run python -m uncertainty_clip.train
```

`utilities/prepare_data.py` reads the downloaded COCO files and writes four CSVs
in `data/openclip/`: train, validation, calibration and test. It uses all 118,287
training images and splits the 5,000 validation images into 1,000 validation,
1,000 calibration and 3,000 test images, using seed 42. Each CSV contains one
fixed caption per image. To download COCO first, use `utilities/download_coco.py`.

`uncertainty_clip/train.py` only calls `open_clip_train.main.main` with the prepared
training and validation CSVs. OpenCLIP owns the model, loader, loss, optimizer,
scheduler, training loop, validation and checkpoints. The default is ViT-B/32,
random initialization, three epochs and one NVIDIA GPU.

Edit the flags in `train.py` or override them on the command line:

```bash
uv run python -m uncertainty_clip.train --epochs 5 --name five-epochs
uv run python -m uncertainty_clip.train --resume latest
```

OpenCLIP saves epoch checkpoints and validation metrics in
`runs/vit-b32/checkpoints/`. To evaluate a checkpoint on validation using OpenCLIP:

```bash
uv run python -m open_clip_train.main \
  --model ViT-B-32 --pretrained runs/vit-b32/checkpoints/epoch_3.pt \
  --val-data data/openclip/validation.csv --dataset-type csv --csv-separator ','
```

Validation uses the one image-caption pair per row. Calibration/test stay unused
until you choose to use them for your uncertainty experiments.
