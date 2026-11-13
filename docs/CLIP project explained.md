# CLIP project explained

We train image and text encoders together from random weights on COCO 2017.
Three ViT-B/32 models use different seeds (42, 43, 44) and no dropout. Their
predictions and learned representations will support deep-ensemble uncertainty
experiments after training quality has been established.

Hugging Face Transformers supplies `CLIPModel`, its symmetric contrastive
loss, `Trainer`, and the native `EarlyStoppingCallback`. Our code prepares the
image-caption CSVs, applies image transforms and computes retrieval recall.
There is no custom optimization loop or early-stopping implementation.

Each batch compares image embeddings against caption embeddings. Correct
image-caption matches lie on the diagonal of the similarity matrix. The
contrastive loss trains both directions, using the other batch members as
negatives. Both encoders produce normalized 512-dimensional embeddings.

The train split has 118,287 images. COCO val2017 is divided deterministically
into 1,000 validation and 4,000 test images. One fixed caption is used per
image. There is no calibration split. Validation drives early stopping and
best-checkpoint selection; test images are reserved for final evaluation.

Training runs for at most 100 epochs and stops after 10 consecutive validation
evaluations without improved loss. Trainer handles checkpoint saving and
restores the selected model. Checkpoints preserve optimizer, scheduler and
callback state for resume. TensorBoard reports training and validation metrics.

See [training commands and persistent sessions](training.md) and the
[research roadmap](roadmap.md). CPU fixture checks validate the pipeline;
full-data GPU performance and useful retrieval still need to be established.
