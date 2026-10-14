# not-a-deepfake: Deepfake Detection in Video Calls

We build a classifier that decides whether the face of a video-call participant is real or a deepfake, using the visual stream only.
## Research questions

1. Which pretrained backbone (CNN or Transformer) gives the best frame-level
   detection, especially on manipulation methods it has not seen in training?
2. Does learned temporal aggregation (LSTM or Transformer) improve on simple
   averaging of frame scores?

## Approach

| Stage | Options compared |
|---|---|
| Frame classifier | Xception, EfficientNet-B4 (with and without Self-Blended Images), ViT-B/16, CLIP ViT-B/16 |
| Temporal aggregation | Mean / top-k pooling, LSTM, Transformer encoder (windows of ~32 frames) |
| Conditions | Clean footage and simulated call quality (compression, low resolution, blur, lighting) |

**Evaluation protocol:** train on FaceForensics++ (c23), then test on unseen
datasets: Celeb-DF v2 (official 518-video test list) and a DFDC subset.

**Metrics:** frame- and video-level AUC, accuracy, F1, TPR at low FPR, and
inference time.


## Data

Datasets are **not** stored in this repository. The FaceForensics++ terms of use
forbid redistribution, and the files are too large for git.

All data lives on the project's Ronin machine, on a separate data drive mounted
at `/mnt/sdd` and linked to `~/data`:

```
~/data/raw/ffpp/      FaceForensics++ c23: original + Deepfakes, Face2Face, FaceSwap, NeuralTextures
~/data/raw/celebdf/   Celeb-DF v2
```

FaceForensics++ splits follow the official train/val/test lists
(3,600 / 700 / 700 videos). There are four fake videos for every real one, so
training should use class weights or balanced sampling.

##
