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

**Evaluation protocol:** train on FaceForensics++ (c23) and Celeb-DF v2, then
test on an unseen dataset: a DFDC subset. Each dataset's own test split is
kept for in-domain results (Celeb-DF: the official 518-video test list).

**Metrics:** frame- and video-level AUC, accuracy, F1, TPR at low FPR, and
inference time.


## Setup

### GPU machine

The GPU machine runs Ronin's Deep Learning GPU (PyTorch) image, which already
has PyTorch installed in `/opt/pytorch`, matched to the GPU driver. Install the
other packages into that environment instead of making a new `.venv`. A new
venv would download its own `torch`, which may not match the driver.

```
git clone https://github.com/rndllchw/not-a-deepfake.git
cd not-a-deepfake
source /opt/pytorch/bin/activate
pip install -r requirements.txt     # skips torch/torchvision, which are already installed
```

So on this machine, the line to run at the start of every session is
`source /opt/pytorch/bin/activate`, not `source .venv/bin/activate`.

If `git clone` asks for a password, log in with `gh auth login` or a GitHub
personal access token typed at the prompt. Never save a token in the repo.


## Data

Datasets are **not** stored in this repository. The FaceForensics++ terms of use
forbid redistribution, and the files are too large for git.

All data lives on the project's Ronin machine, on a separate data drive mounted
at `/mnt/sdd` and linked to `~/data`.

### Raw data layout

```
~/data/raw/
├── ffpp/                                   FaceForensics++, as downloaded
│   ├── original_sequences/youtube/c23/videos/
│   │   └── 000.mp4 … 999.mp4               1,000 real videos, one person each
│   └── manipulated_sequences/
│       ├── Deepfakes/c23/videos/           1,000 fakes per method
│       ├── Face2Face/c23/videos/
│       ├── FaceSwap/c23/videos/
│       └── NeuralTextures/c23/videos/
│           └── 000_003.mp4 …               fake made from real videos 000 and 003
└── celebdf/                                Celeb-DF v2
    ├── Celeb-real/
    │   └── id0_0000.mp4 …                  590 real clips of 59 celebrities (person 0, clip 0000)
    ├── YouTube-real/
    │   └── 00000.mp4 …                     300 extra real clips of other people from YouTube
    ├── Celeb-synthesis/
    │   └── id0_id16_0000.mp4 …             5,639 fakes, faces swapped between celebrities 0 and 16
    └── List_of_testing_videos.txt          official 518-video test list ("1" = real, "0" = fake)
```

`c23` is FF++'s high-quality compression level; FF++ also comes as `raw`
(uncompressed) and `c40` (heavily compressed). The YouTube-real clips have no
person ids and are never used to make fakes, so each counts as its own person.

### Splits

Videos are split **by person**, so nobody seen in training appears in
validation or test. Otherwise a model can score well by recognising faces and
backgrounds instead of detecting manipulation.

- **FaceForensics++:** the official train/val/test lists in `splits/ffpp/`
  (3,600 / 700 / 700 videos). Each list entry is a pair of ids, e.g.
  `["000", "003"]`, and puts `000.mp4`, `003.mp4`, `000_003.mp4` and
  `003_000.mp4` (in all four methods) into the same split.
- **Celeb-DF:** the official 518-video list is the test split. The remaining
  videos go to train or val: 10% of the celebrities and 10% of the YouTube clips
  (chosen with a fixed seed) go to val, everything else to train. Celeb-DF test
  videos share people with train, so treat Celeb-DF test results as in-domain.

**Dropped videos.** A Celeb-DF fake shows two people. If one of them is in
train and the other in val, the fake would put a person on both sides whichever
split it went to, so it is dropped. With 10% of people in val (seed 0) this
removes 1,072 of the 5,299 non-test fakes (20%). Real videos are never dropped.
`build_index` prints the exact number.

Both datasets have far more fake than real videos (FF++ 4:1, Celeb-DF about
6:1), so training should use class weights or balanced sampling.

### Video index

Every later stage reads its list of videos from one index file, with one row
per video. Build it on Ronin from the repo root:

```
python -m src.preprocessing.build_index --data-root ~/data   # writes ~/data/processed/index.csv
```

| Column | Type | Example values | Meaning |
|---|---|---|---|
| `video_id` | string | `ffpp/real/000`, `ffpp/Deepfakes/000_003`, `celebdf/YouTube-real/00170` | Unique key, `dataset/method/name` |
| `dataset` | string | `ffpp`, `celebdf` | Source dataset |
| `method` | string | `real`, `Deepfakes`, `Face2Face`, `FaceSwap`, `NeuralTextures`, `Celeb-real`, `YouTube-real`, `Celeb-synthesis` | How the video was made (Celeb-DF: folder name) |
| `label` | int | `0`, `1` | 0 = real, 1 = fake |
| `split` | string | `train`, `val`, `test` | Celeb-DF: official list is `test`, the rest is split by person into `train`/`val`   |
| `source_ids` | string | `000`, `000_003`, `id0_id16`, `yt00170` | People in the video, used to check that no identity crosses splits |
| `path` | string | `/home/<user>/data/raw/ffpp/.../000.mp4` | Full path to the video file (`~` expanded) |

Read it with `pd.read_csv(path, dtype={"source_ids": str})` so IDs such as
`000` keep their leading zeros.

### Face crops

Deepfakes only change the face, and the backbones take 224–380 px inputs, so
every model trains on face crops rather than whole frames. Extract them once on
the GPU machine after building the index (in tmux):

```
python -m src.preprocessing.extract_faces --data-root ~/data --limit 20   # quick test, prints s/video
python -m src.preprocessing.extract_faces --data-root ~/data              # everything
```

- **Frames:** 2 windows of 32 consecutive frames per video, one in the middle
  of each half (`--windows`, `--window-len`). The temporal models use the
  windows; the frame models use the same crops. Videos shorter than two
  windows get one.
- **Faces:** detected with RetinaFace (ResNet-50, via `batch-face`), a whole
  window at a time, on frames shrunk to 960 px; crops come from the full frame.
  Each window starts with the largest face and follows it frame to frame, so a
  window stays on one person; different windows may pick different people.
- **Landmarks:** RetinaFace's 5 points (eyes, nose, mouth corners) go in the
  CSV. The standard 68 points (FAN, via `face-alignment`; needed for
  Self-Blended Images masks) go in `landmarks68.npy` next to the crops.
- **Crops:** square, 1.3× the face box (`--margin 0.3`), resized to 380 px
  (`--size`), JPEG quality 95. Parts outside the frame are black. Resize per
  backbone when loading (Xception 299, EfficientNet-B4 380, ViT/CLIP 224).
- **Resumable:** each video's `faces.csv` is written last; videos that have one
  are skipped, so the job can be stopped with Ctrl+C and rerun. Videos that
  fail are reported and retried on the next run.
- **GPU:** `--workers` processes (default 4) decode videos on the CPU, and each
  loads its own copy of the models on the GPU. Lower it if the GPU runs out of
  memory; raise it if the GPU sits idle and there are spare CPUs.
  `--device cpu` works for testing but is slow (~40 s/video).

Output in `~/data/faces/<video_id>/`: one `<frame>.jpg` per frame with a face,
`faces.csv`, and `landmarks68.npy`. All videos' rows are also merged into
`~/data/processed/faces.csv`, one row per extracted frame:

| Column | Meaning |
|---|---|
| `video_id`, `dataset`, `method`, `label`, `split` | As in the index |
| `window`, `frame` | Window number in the video (0, 1) and frame number in the video |
| `crop` | Crop path relative to `~/data/faces/`; empty if no face was found in that frame |
| `score` | RetinaFace face confidence |
| `crop_x`, `crop_y`, `crop_side` | Where the crop sits in the original frame (pixels) |
| `right_eye_x` … `mouth_left_y` | 5 face landmarks in crop pixels (the person's right eye first) |
| `lm68_score` | Mean confidence of the 68 landmarks (0–1) |

`landmarks68.npy` has shape `(rows, 68, 2)`: the 68 points (iBUG order: jaw
0–16, brows 17–26, nose 27–35, eyes 36–47, mouth 48–67) in crop pixels, one
entry per row of that video's `faces.csv`, in the same order; NaN where no face
was found.


## Frame models

Code in `src/frames/`. Each backbone is fine-tuned on face crops to output a
fake probability per crop and a feature vector for the temporal models.

| `--backbone` | timm weights | Input | Features |
|---|---|---|---|
| `xception` | `legacy_xception.tf_in1k` | 299 | 2048 |
| `efficientnet_b4` | `tf_efficientnet_b4.ap_in1k` | 380 | 1792 |
| `vit_b16` | `vit_base_patch16_224.augreg_in21k` | 224 | 768 |
| `clip_vit_b16` | `vit_base_patch16_clip_224.openai` | 224 | 768 |

- **Quality filter:** crops with a RetinaFace score below 0.9 or a crop side
  under 64 px in the original frame are left out (`--min-score`, `--min-side`).
- **Balanced sampling:** each epoch draws real and fake equally often, and
  every video equally often within its class.
- **Call quality:** half the training crops (`--degrade-prob`) get random
  lighting, noise, blur, downscaling and strong JPEG compression
  (`src/frames/degrade.py`). `predict --call-quality` applies the same to the
  test crops, identically for every model.
- **Self-Blended Images** (`--sbi only` or `--sbi mix`): real crops blended with
  an altered copy of themselves inside a soft face mask, labelled fake
  (`src/frames/sbi.py`). `only` trains on real crops and SBI fakes alone, as in
  the paper.
- **Which weights train** (`--train-mode`): `full`, `linear` (head only), or
  `norm` (head and normalisation layers; the light CLIP fine-tune).
- **Unseen methods:** `--exclude-methods NeuralTextures` leaves a method out of
  train and val; `predict` still scores it.

### Input setting: mask vs no-mask

Every model is trained in one of two input settings, chosen with `--mask`:

| Setting | Flag | The model sees |
|---|---|---|
| `no-mask` | (none) | the plain face crop, with hair, neck and background |
| `mask` | `--mask` | only the face: outside the 68-landmark outline (extended over the forehead, plus a small border) is black |

All masking code is in `src/frames/mask.py`, applied in one marked step of
`FaceFrames.__getitem__` (`src/frames/dataset.py`). The setting is part of the
run name and folder, and `predict` reads it from the run, so a model is always
tested the way it was trained. Preview the mask on a few crops:
`python -m src.frames.mask --data-root ~/data` (writes `~/data/previews/mask/`).

### Running

Baselines (the four backbones, each with and without the mask; 8 runs), in
tmux on the GPU machine:

```
bash scripts/run_frame_baselines.sh --epochs 1 --samples-per-epoch 256 --val-frames 1  # quick test
bash scripts/run_frame_baselines.sh
```

Each run trains, then scores the test split clean and at call quality. The
script is safe to rerun after the machine stops: finished runs are skipped and
unfinished ones continue. It ends by printing the comparison table
(`python -m src.frames.compare`, also saved as `~/data/runs/frames/comparison.csv`).

Single runs:

```
python -m src.frames.train --backbone efficientnet_b4            # no-mask
python -m src.frames.train --backbone efficientnet_b4 --mask     # mask
python -m src.frames.train --backbone efficientnet_b4 --sbi only
```

Runs are saved by input setting:

```
~/data/runs/frames/
├── no-mask/<run name>/      best.pt, last.pt, metrics.csv, predictions/
├── mask/<run name>/
└── comparison.csv
```

If the machine stops mid-run, rerun the same command with `--resume`. To score
again or score other splits, use `python -m src.frames.predict --run <run folder>`;
for the temporal models, save features for every split with
`predict --split train --features` (and `val`, `test`).
