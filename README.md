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
python -m src.data.build_index --data-root ~/data   # writes ~/data/processed/index.csv
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
