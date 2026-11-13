"""Train the three no-dropout ensemble members with OpenCLIP's trainer."""

import os
from pathlib import Path
import subprocess
import sys


def main():
    common = [
        "--epochs", "100",
        "--logs", os.environ.get("CLIP_LOGS", "runs/openclip-suite"),
        "--resume", "latest",
        "--report-to", "tensorboard",
    ]
    for seed in (42, 43, 44):
        subprocess.run(
            [sys.executable, "-m", "uncertainty_clip.train", *common,
             "--name", f"baseline-{seed}", "--seed", str(seed), *sys.argv[1:]],
            cwd=Path(__file__).resolve().parents[1], check=True,
        )


if __name__ == "__main__":
    main()
