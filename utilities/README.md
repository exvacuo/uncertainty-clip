# Dataset utilities

## Download COCO 2017

Use the project's `uv` environment and have `curl` installed. The script uses
only the Python standard library. Run these commands from the repository root.

```bash
# Download train and validation images, plus captions (default).
uv run utilities/download_coco.py

# Download validation images first for pipeline development.
uv run utilities/download_coco.py --split val

# Add training images later.
uv run utilities/download_coco.py --split train

# Print URLs and destinations without downloading or creating files.
uv run utilities/download_coco.py --dry-run

# Override the dataset root, for example on a GPU server.
uv run utilities/download_coco.py --data-dir /scratch/datasets/coco
```

The default location is this repository's `data/coco/`, even when the script is
invoked from another working directory:

```text
data/coco/
├── images/
│   ├── train2017/
│   └── val2017/
├── annotations/
│   ├── captions_train2017.json
│   └── captions_val2017.json
└── .downloads/                 # Partial downloads and completion markers
```

Both caption files are extracted from the shared official annotations archive,
even when downloading only one image split. Detection, segmentation, and
keypoint annotations are not extracted.

The complete download transfers roughly 19–20 GB. Allow about 40 GB of free disk
space for archives and extracted images during setup. Rerunning the command
resumes partial downloads and skips archives already extracted by this script.
Completion markers are not a full integrity check of previously extracted data.
If you remove or modify dataset files, remove the corresponding `.done` marker
in `.downloads/` before rerunning to restore that archive.

ZIP files are removed after successful extraction; pass `--keep-archives` to
retain them. Interrupted extraction can be retried from the cached ZIP. A ZIP
integrity error leaves the archive available for inspection; remove that archive
to download it again.

Downloads use the URLs published on the [official COCO download page](https://cocodataset.org/#download).
After downloading, prepare all image-caption CSV files with the single preparation script:

```bash
uv run utilities/prepare_data.py
```

This writes `train.csv`, `validation.csv`, `calibration.csv` and `test.csv` in
`data/openclip/`. Then run `uv run python -m uncertainty_clip.train`.
