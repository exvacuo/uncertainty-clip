# CNN image encoders for CLIP

Research date: 2026-10-05. Scope: selecting a model for training both encoders
from random initialization, likely on COCO, before uncertainty experiments.

## Findings

CNN image encoders are an established CLIP option. The original work trained
five ResNet variants alongside ViTs. Its modified ResNet includes an attention
pooling head, so OpenCLIP's `RN50` is not identical to torchvision's ordinary
ResNet-50. See [CLIP, Section 2.3 and Table 11](https://arxiv.org/pdf/2103.00020).

| Model or study | Training data | Reported ImageNet zero-shot top-1 |
| --- | --- | --- |
| Original CLIP RN50 | WIT, 400M pairs | 59.6% |
| Original CLIP RN50x64 | WIT, 400M pairs | 73.6% |
| Original CLIP ViT-B/32, for context | WIT, 400M pairs | 63.2% |
| DeCLIP RN50 | 56M pairs | 60.4% |
| OpenCLIP ConvNeXt-Base with augmentation/regularization | LAION-2B, 13B sample presentations | 71.5% |

The first three rows come from CLIP Table 11. These results show CNN viability;
they do not establish which architecture works best on COCO.

[DeCLIP](https://arxiv.org/pdf/2110.05208) adds self-supervision, multiple views,
and nearest-neighbor supervision. Its result uses a different objective and
data mixture, so it is not an isolated CNN-versus-ViT comparison.

The [ConvNeXt model card](https://huggingface.co/laion/CLIP-convnext_base_w-laion2B-s13B-b82K)
documents the CNN models and training setup. Its 71.5% figure belongs to the
`augreg` variant; the ordinary 256-pixel variant scores 70.8%. Sample
presentations include repeated visits to training pairs.

## Evidence closer to COCO

### Published work: CLIP-Lite, AISTATS 2023

[CLIP-Lite](https://proceedings.mlr.press/v206/shrivastava23a/shrivastava23a.pdf)
studies ResNet-50 with COCO captions. Table 1 reports ImageNet linear-classifier
accuracy of 33.2% for its CLIP-COCO baseline and 55.3% for CLIP-Lite. These
scores involve a classifier trained with ImageNet labels; they are not zero-shot
results. The main CLIP-Lite setup initializes its text encoder from BERT.
Table 10 separately tests randomly initialized text Transformers: two and twelve
layers reach 87.5 and 87.9 VOC07 mAP. CLIP-Lite changes the objective, so it is
related evidence rather than a direct reproduction of our planned baseline.

### Independent implementation: revantteotia/clip-training

The [author's repository](https://github.com/revantteotia/clip-training) describes
training on COCO with a ResNet-50, a six-layer text Transformer, batch size 256,
and four GTX 1080 GPUs.

Its [evaluation report](https://github.com/revantteotia/clip-training/blob/main/zero_shot_eval_output/coco_trained_clip_observations.md)
shows qualitative image–text alignment but weak broad zero-shot classification:
4.09% ImageNet top-1 and 15.07% CIFAR-10 top-1. These are author-reported results,
not independently reproduced here. Model initialization could not be fully
verified from the accessible model files, so this should not be cited as a
verified reproduction of both encoders trained from scratch.

## Original recommendation for this project

**Update, 2026-10-13:** the project now selects ViT-B/32 as its first baseline.
The RN50 recommendation below records the earlier rationale; neither
architecture has yet been shown to be better on our COCO setup.

Prioritize OpenCLIP's modified ResNet-50 (`RN50`) for the first pilot, keeping
ViT-B/32 as a comparison. Randomly initialize both encoders and retain the
standard symmetric CLIP loss. This is a practical research recommendation,
not a finding that RN50 is better on COCO.

CNNs encode assumptions about nearby pixels and reuse filters across image
positions. The [original ViT paper](https://arxiv.org/pdf/2010.11929) discusses
how these assumptions can help with limited data. Applying that argument to
our COCO contrastive setup is a hypothesis; training recipe and model size
also affect the outcome.

For the first comparison:

1. Use identical image splits, caption sampling, evaluation candidate pools,
   and the same contrastive loss.
2. Keep the text encoder and shared embedding dimension fixed when isolating
   the image architecture. OpenCLIP's stock RN50 and ViT-B/32 configurations
   have different embedding dimensions, so a stock comparison evaluates whole
   configurations rather than only the image backbone.
3. Measure held-out retrieval Recall@1/5/10, overfitting, GPU memory, and
   throughput. Report both training exposure and GPU time.
4. Tune learning rates fairly for each model. Repeat promising settings with
   multiple seeds before choosing the uncertainty-study baseline.
5. Treat reducing text depth as a separate experiment; changing both encoders
   at once prevents attributing a difference to the CNN.

Start with the standard CLIP text Transformer for a reference run. A smaller
text Transformer is worth investigating if the pilot shows overfitting or
excessive training cost.

For later dropout experiments, account for ResNet's batch-normalization layers:
enable only the intended stochastic layers during sampling and keep batch
normalization in evaluation mode. Add and train dropout deliberately; switching
the entire model to training mode is not a valid shortcut.

## Implementation references

- [OpenCLIP RN50 configuration](https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/model_configs/RN50.json)
- [OpenAI reference model, including ModifiedResNet](https://github.com/openai/CLIP/blob/main/clip/model.py)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [CLIP-Lite author implementation](https://github.com/4m4n5/CLIP-Lite)

No model training or benchmark reproduction was performed for this review.
