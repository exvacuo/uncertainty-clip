"""Run with python -m uncertainty_clip {prepare,train,evaluate,benchmark}."""

import argparse
import json
from pathlib import Path

from .config import SUPPORTED_MODELS, TrainConfig
from .data import prepare_splits


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train CLIP from scratch and evaluate COCO retrieval")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Verify COCO files and save disjoint image splits")
    prepare.add_argument("--data-dir", default="data/coco")
    prepare.add_argument("--output", default="data/splits/coco2017.json")
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--validation-size", type=int, default=1000)
    prepare.add_argument("--calibration-size", type=int, default=1000)

    train = commands.add_parser("train", help="Train both encoders from random initialization")
    train.add_argument("--config", default="configs/vit-b32.toml")
    train.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"))
    train.add_argument("--data-dir")
    train.add_argument("--manifest")
    train.add_argument("--output", dest="output_dir")
    train.add_argument("--resume", action="store_true", help="Resume last.pt in the configured run directory")

    evaluate = commands.add_parser("evaluate", help="Evaluate a saved checkpoint against all captions")
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "calibration", "test"), default="validation")
    evaluate.add_argument("--data-dir")
    evaluate.add_argument("--manifest")
    evaluate.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--limit", type=int, default=0, help="Fixed image subset; 0 evaluates the whole split")
    evaluate.add_argument("--output", help="Optional new JSON result file")

    benchmark = commands.add_parser("benchmark", help="Estimate training time and GPU memory for CLIP models")
    benchmark.add_argument("--config", default="configs/vit-b32.toml")
    benchmark.add_argument("--models", nargs="+", choices=SUPPORTED_MODELS, default=list(SUPPORTED_MODELS))
    benchmark.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    benchmark.add_argument("--data-dir")
    benchmark.add_argument("--manifest")
    benchmark.add_argument("--batch-size", type=int)
    benchmark.add_argument("--workers", type=int)
    benchmark.add_argument("--epochs", type=int, help="Epochs to project for the configured run")
    benchmark.add_argument("--precision", choices=("auto", "fp32", "amp"))
    benchmark.add_argument("--warmup-steps", type=int, default=5)
    benchmark.add_argument("--steps", type=int, default=30, help="Measured training updates per model")
    benchmark.add_argument("--output", help="Optional new JSON report, e.g. runs/compute-estimate.json")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_splits(args.data_dir, args.output, args.seed, args.validation_size, args.calibration_size)
            print(json.dumps({"manifest": str(Path(args.output).resolve()),
                              "images": {name: len(ids) for name, ids in manifest["splits"].items()}}, indent=2))
        elif args.command == "train":
            config = TrainConfig.load(args.config, device=args.device, data_dir=args.data_dir,
                                      manifest=args.manifest, output_dir=args.output_dir)
            from .training import train as run_training

            run_training(config, resume=args.resume)
        elif args.command == "benchmark":
            from .benchmark import benchmark_models

            config = TrainConfig.load(args.config, device=args.device, data_dir=args.data_dir,
                                      manifest=args.manifest, batch_size=args.batch_size, workers=args.workers,
                                      epochs=args.epochs, precision=args.precision)
            report = benchmark_models(config, args.models, args.warmup_steps, args.steps, args.output)
            if any(result["status"] != "ok" for result in report["results"]):
                parser.exit(1, "Some models failed the benchmark; inspect the report before budgeting a run.\n")
        else:
            from .training import evaluate_checkpoint

            evaluate_checkpoint(args.checkpoint, args.split, args.data_dir, args.manifest, args.device,
                                args.batch_size, args.workers, args.limit, args.output)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Missing dependency {error.name!r}. Install the environment with uv sync.\n")
    except (ValueError, TypeError, KeyError, OSError, FloatingPointError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
