# Research roadmap

Study uncertainty estimation for CLIP predictions. Both image and text encoders
must start from random weights. The current baseline is OpenCLIP 3.2.0's
ViT-B/32 trained on COCO 2017. Implementation and CPU pipeline checks are
complete; GPU experiments remain pending. See [the training guide](training.md)
for commands and the fixed image-disjoint split protocol.

## Establish the GPU budget

Use OpenCLIP's training logs on the intended GPU to measure throughput and check
memory use for ViT-B/32, ViT-B/16 and RN50. Record batch size, precision and device.
Identify a feasible batch size before launching a full run.

Exact university GPU allocation and job-submission requirements still need to
be established with Matias.

## Train and select the baseline

Run ViT-B/32 and inspect OpenCLIP's validation retrieval metrics. Check finite loss, skipped updates,
overfitting, throughput, memory, and checkpoint/resume behavior on CUDA.

Choose training duration and learning rate using validation results. The default three-epoch recipe is a starting point, not evidence of convergence.

If budget permits, compare ViT-B/16 and RN50 using identical splits, caption
sampling, and retrieval candidate pools. Tune learning rates fairly and report
both training exposure and GPU time. Repeat promising configurations with
multiple seeds. Stock RN50 and ViT models differ in text width and embedding
dimension, so that comparison evaluates complete configurations. Isolating the
image backbone would require matching those components.

COCO is much smaller than the original CLIP training corpus. Establish useful
held-out retrieval before relying on the model for uncertainty experiments.
This is a custom COCO 2017 protocol, not the Karpathy retrieval benchmark.

## Uncertainty experiments

After selecting a trained baseline:

1. Define the prediction task and fixed candidate pool. Retrieval and
   prompt-based classification need distinct evaluation protocols.
2. Measure maximum softmax probability and predictive entropy.
3. Fit temperature scaling on the reserved calibration images. Keep this
   temperature distinct from CLIP's learned training logit scale.
4. Study prompt variation for the chosen classification task.
5. Train independent models with different seeds for an ensemble comparison.
6. Specify and train a dropout variant if investigating stochastic predictions.

Calling `model.train()` alone does not define a dropout experiment. Specify
locations, rates, and training procedure; for RN50, keep batch normalization
in evaluation mode during stochastic inference.

Keep calibration separate from model selection. Use the final test split only
after the recipe and uncertainty protocol are fixed. Record configuration,
software versions, source revision, seeds, data hashes, and checkpoint selection.

## References

- [Original CLIP paper](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf).
- [Pinned OpenCLIP release](https://github.com/mlfoundations/open_clip/releases/tag/v3.2.0).
- [Archived CNN encoder review](archive/cnn-encoder-research.md).
