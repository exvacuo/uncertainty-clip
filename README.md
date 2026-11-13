# Uncertainty Quantification with CLIP

This project investigates uncertainty quantification in Contrastive Language–Image Pre-training (CLIP) models.

The objective is to train multiple CLIP models and explore different approaches to estimating uncertainty, including deep ensembles, Monte Carlo (MC) Dropout, and methods based on the variability and distribution of learned representations.

The implementation uses OpenCLIP 3.2.0 and PyTorch 2.8.0 for model training and evaluation.

## 1. Model Architecture

The experiments use the CLIP ViT-B/32 architecture, consisting of:

- **Image encoder:** Vision Transformer with 12 layers.
- **Text encoder:** Transformer with 12 layers.
- **Embedding dimension:** 512 for both image and text representations.
- **Initialization:** Random initialization (no pretrained weights).
- **Training:** All model parameters are trainable.

Both encoders produce L2-normalized embeddings in a shared representation space.

### Model Configurations

Three models are trained independently:

| Model | Random Seed | Dropout | Purpose |
|---|---:|---|---|
| CLIP 1 | 42 | None | Deep ensemble |
| CLIP 2 | 43 | None | Deep ensemble |
| CLIP 3 | 44 | None | Deep ensemble |

The first three models have identical architectures and training configurations but differ in their random initialization seeds.

Together, they form a deep ensemble, allowing uncertainty to be investigated through differences in their predictions and learned representations.

The launcher trains only these three models, without patch, attention or MLP dropout.
A separate patch-dropout experiment is deferred.

## 2. Dataset

Training uses the COCO 2017 dataset.

The dataset is divided as follows:

| Split | Images | Purpose |
|---|---:|---|
| Training | 118,287 | Model optimization |
| Validation | 1,000 | Validation and manual checkpoint selection |
| Test | 4,000 | Reserved for final evaluation |

The original COCO validation set contains 5,000 images and is divided into validation and test subsets using random seed 42.

Each image is associated with one fixed caption selected during dataset preparation.

### Data Preparation

Download the COCO dataset using:

```bash
uv run utilities/download_coco.py
```

Prepare the dataset:

```bash
uv run utilities/prepare_data.py
```

This generates three CSV files (`train.csv`, `validation.csv`, and `test.csv`) in:

```text
data/openclip/
```

The generated files contain the image paths and corresponding captions for each split.

### Preprocessing

Training images undergo:

- Random resized cropping to 224 × 224.
- Bicubic interpolation.
- RGB conversion.
- CLIP image normalization.

Validation images undergo resizing followed by a 224 × 224 center crop.

Text is tokenized using OpenCLIP’s BPE tokenizer with a maximum context length of 77 tokens.

## 3. Training Configuration

All three runs call our `uncertainty_clip.train` wrapper, which passes the shared training configuration to OpenCLIP's official trainer. Each run uses a different random seed.

| Parameter | Value |
|---|---|
| Architecture | ViT-B/32 |
| Initialization | Random |
| Dataset | COCO 2017 |
| Batch size | 128 |
| Optimizer | AdamW |
| Learning rate | 5 × 10⁻⁴ |
| Weight decay | 0.2 |
| LR scheduler | Linear warmup + cosine decay |
| Warmup | 20 optimizer steps |
| Maximum epochs | 100 |
| Gradient clipping | 1.0 |
| Mixed precision | CUDA AMP |
| Early stopping | None; runs to the requested epoch limit |
| Checkpoints | Every epoch and latest; manual selection using validation |

### Training Objective

Models are optimized using the symmetric CLIP contrastive loss.

The objective combines:

- Image-to-text contrastive cross-entropy.
- Text-to-image contrastive cross-entropy.

In-batch negatives are used during training.

### Validation and Checkpoints

OpenCLIP evaluates validation after each epoch and records retrieval Recall@1, @5 and @10 in both directions, plus validation loss. It saves numbered epoch checkpoints and `epoch_latest.pt`. Select a checkpoint manually using validation before evaluating on test data. There is no custom early stopping or automatic best-checkpoint selection.

The stock CSV training loader shuffles data and drops the incomplete final batch. At batch size 128, it uses 118,272 of 118,287 training pairs each epoch; the omitted pairs can differ across epochs.

## 4. Running the Experiments

### Installation

Install the project dependencies using:

```bash
uv sync --locked
```

### Training the Three Models Without Dropout

To train all three ensemble members:

```bash
uv run python scripts/train_models.py --epochs 100
```

Prepare the dataset first using the commands above. The Python script then calls our training wrapper three times sequentially, using seeds 42, 43 and 44. The wrapper invokes the official OpenCLIP trainer. No dropout run is launched. Additional arguments are forwarded to OpenCLIP; keep the model, seed, run name and dropout settings unchanged for this ensemble.

Training outputs are stored in:

```text
runs/openclip-suite/
```

OpenCLIP resumes each run from its latest checkpoint when the same command is repeated. To move computers, copy the run directory (including optimizer checkpoints), use matching dependencies and training flags, regenerate CSV paths, and set `CLIP_LOGS` to the copied suite directory. Use a new directory for different training settings. The earlier custom suite and its checkpoints remain separate and require their original code to resume.

### Training a Single Model

Individual training runs can be launched using:

```bash
uv run python -m uncertainty_clip.train
```

The default single-model configuration uses ViT-B/32 with random initialization and three epochs.

To override the number of epochs:

```bash
uv run python -m uncertainty_clip.train --epochs 5 --name five-epochs
```

To resume training from the latest checkpoint:

```bash
uv run python -m uncertainty_clip.train --resume latest
```

## 5. Training Time Estimation

An initial timing experiment was performed on a local GPU.

The measured training time was approximately:

**5 minutes and 12 seconds per epoch.**

Based on this measurement:

| Configuration | Estimated Training Time |
|---|---|
| 1 epoch | 5 min 12 s |
| 100 epochs (one model) | 8 h 40 min |
| 100 epochs (three models) | 26 h |

These estimates assume sequential execution and the same approximate epoch duration.

The actual total runtime may differ due to validation, checkpoint saving, and other overhead.

These runs have no early stopping; they train to the requested epoch limit.

## 6. Validation and Evaluation

Model selection is based on retrieval performance on the validation subset.

The evaluation metrics include:

- Recall@1
- Recall@5
- Recall@10

Metrics are calculated for both image-to-text and text-to-image retrieval.

Epoch checkpoints are retained; select a checkpoint using validation before final test evaluation.

The test dataset remains separate from training and checkpoint selection and is reserved for final evaluation.

## 7. Uncertainty Quantification

The next stage of the project involves investigating uncertainty quantification methods using the trained CLIP models and their learned representations.

The following approaches are being considered.

### 7.1. Deep Ensembles

The three independently initialized CLIP models can be used to investigate uncertainty through differences between their predictions or representations.

Potential analyses include:

- Variability between model predictions.
- Variability in image and text embeddings.
- Agreement and disagreement between ensemble members.

### 7.2. Monte Carlo Dropout

A future separate model could use OpenCLIP’s native image patch dropout with probability 0.1. The training launcher does not include this experiment.

By enabling only the image patch-dropout module during inference and performing multiple forward passes, it is possible to obtain a distribution of predictions or embeddings from a single trained model.

Potential analyses include:

- Variability across stochastic forward passes.
- Predictive uncertainty.
- Embedding variability.

OpenCLIP’s patch-dropout module can be enabled for stochastic image embeddings. Text embeddings remain deterministic. Repeated-sampling and aggregation procedures remain to be implemented.

### 7.3. Feature-Based Uncertainty

Additional uncertainty estimation methods will be investigated using CLIP’s learned feature representations.

In particular:

**Feature variability**

Analyze variations in image and text embeddings across different ensemble members or stochastic forward passes.

**Mahalanobis distance**

Investigate uncertainty estimation using the distance between feature representations and reference feature distributions, taking feature covariance into account.

**KL divergence**

Explore the use of Kullback–Leibler divergence to quantify differences between suitable probability distributions derived from model outputs or representations.

The precise definitions of the distributions, uncertainty scores, and evaluation procedures remain to be determined.
