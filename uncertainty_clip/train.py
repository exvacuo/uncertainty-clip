"""Train using the prepared CSVs and OpenCLIP's official trainer."""

import sys
from open_clip_train.main import main

# Runs the training script from OpenCLIP with the specified arguments.
if __name__ == "__main__":
    main([
        "--train-data", "data/openclip/train.csv",
        "--val-data", "data/openclip/validation.csv",
        "--dataset-type", "csv",
        "--csv-separator", ",",
        "--csv-img-key", "filepath",
        "--csv-caption-key", "title",
        "--model", "ViT-B-32",
        "--batch-size", "128",
        "--epochs", "3",
        "--lr", "5e-4",
        "--wd", "0.2",
        "--warmup", "20",
        "--grad-clip-norm", "1.0",
        "--workers", "4",
        "--device", "cuda",
        "--precision", "amp",
        "--seed", "42",
        "--logs", "runs",
        "--name", "vit-b32",
        "--save-frequency", "1",
        "--save-most-recent",
    ] + sys.argv[1:])
