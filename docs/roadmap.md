# Research roadmap

Train three randomly initialized Transformers CLIP ViT-B/32 models on COCO
2017 with seeds 42, 43 and 44, without dropout. Use native Trainer early
stopping and select models using the 1,000-image validation split. The
4,000-image test split is reserved for final evaluation. There is no separate
calibration set.

## GPU pilot

Measure memory, throughput and validation behavior on the workstation before
committing the full budget. Check finite loss and updates in both towers,
checkpoint resume and TensorBoard logging. Keep the epoch limit and other
settings fixed when resuming so the cosine schedule remains consistent.

The default learning rate, 20-step warmup, batch size 128, 100-epoch ceiling
and patience 10 are starting settings, not evidence of convergence. Tune on
validation only and record code revision, dependencies, seeds, data hashes,
training exposure and GPU time. COCO is much smaller than the original CLIP
training corpus; establish useful held-out retrieval before interpreting
uncertainty scores.

## Ensemble experiments

After selecting the three trained models:

1. Define the prediction task and a fixed candidate pool.
2. Evaluate retrieval and agreement across ensemble members.
3. Compare predictive entropy, confidence and embedding variability.
4. If fitting confidence calibration, use non-test data under an explicit
   fitting protocol. Keep its temperature distinct from CLIP's trained logit scale.
5. Evaluate on test only after fixing the recipe and uncertainty protocol.

MC dropout is outside the current implementation scope. Any later dropout
experiment needs an explicit choice of locations, rates and training and
inference behavior.

See [the training guide](training.md). Historical encoder research is retained
in [the archive](archive/cnn-encoder-research.md).
