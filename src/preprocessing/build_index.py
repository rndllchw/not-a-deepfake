"""
Builds the video index with labels and splits.
One row per video.

Only reads file names, not video contents.
Run on the Ronin machine:
    python -m src.preprocessing.build_index --data-root ~/data

Expected layout from official download scripts:
    {data_root}/raw/ffpp/original_sequences/youtube/c23/videos/000.mp4
    {data_root}/raw/ffpp/manipulated_sequences/<Method>/c23/videos/000_003.mp4
    {data_root}/raw/celebdf/{Celeb-real,Celeb-synthesis,YouTube-real}/*.mp4
    {data_root}/raw/celebdf/List_of_testing_videos.txt

FF++ split files (train/val/test.json from the FaceForensics GitHub repo) are
lists of video id pairs whose face swaps belong together, and live in this repo
under splits/ffpp/.

Celeb-DF: the official test list is the test split; the other videos are split
by person into train and val (see index_celebdf).
"""

import argparse
import json
import random
import re
from pathlib import Path

import pandas as pd

FFPP_METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
SPLITS = ["train", "val", "test"]
CELEBDF_FOLDERS = ["Celeb-real", "YouTube-real", "Celeb-synthesis"]
REPO_ROOT = Path(__file__).resolve().parents[2]


def require_videos(folder: Path) -> list[Path]:
    """All .mp4 files in a folder; fails clearly if the folder is missing or empty."""
    videos = list(folder.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 files in {folder}; check --data-root and --compression")
    return videos


def load_ffpp_splits(splits_dir: Path) -> dict[str, str]:
    """Map each FF++ id ("000") and pair ("000_003") to its split."""
    id_to_split = {}
    for split in SPLITS:
        pairs = json.loads((splits_dir / f"{split}.json").read_text())
        # e.g. [["071", "054"], ["087", "081"], ...]
        for a, b in pairs:
            for key in (a, b, f"{a}_{b}", f"{b}_{a}"):
            # Iterate over all 4 permutable video names
                if id_to_split.get(key, split) != split:
                    raise ValueError(f"{key} appears in both {id_to_split[key]} and {split}")
                id_to_split[key] = split
    return id_to_split


def index_ffpp(ffpp_root: Path, splits_dir: Path, compression: str) -> pd.DataFrame:
    id_to_split = load_ffpp_splits(splits_dir)
    rows = []

    real_dir = ffpp_root / "original_sequences" / "youtube" / compression / "videos"
    for path in sorted(require_videos(real_dir)):
        rows.append(dict(video_id=f"ffpp/real/{path.stem}", dataset="ffpp", method="real",
                         label=0, split=id_to_split.get(path.stem), source_ids=path.stem,
                         path=str(path)))

    for method in FFPP_METHODS:
        fake_dir = ffpp_root / "manipulated_sequences" / method / compression / "videos"
        for path in sorted(require_videos(fake_dir)):
            rows.append(dict(video_id=f"ffpp/{method}/{path.stem}", dataset="ffpp", method=method,
                             label=1, split=id_to_split.get(path.stem), source_ids=path.stem,
                             path=str(path)))

    df = pd.DataFrame(rows)
    missing = df["split"].isna()
    if missing.any():
        print(f"[ffpp] warning: {missing.sum()} videos not in any split, dropped: "
              f"{df.loc[missing, 'video_id'].head().tolist()} ...")
        df = df[~missing]
    return df


def celebdf_identities(stem: str) -> list[str]:
    """People in a Celeb-DF video: id0_0000 -> [id0], id0_id16_0000 -> [id0, id16].
    YouTube-real videos (00170) have no celebrity id, so each is its own person."""
    return re.findall(r"id\d+", stem) or [f"yt{stem}"]


def index_celebdf(celebdf_root: Path, val_frac: float, seed: int) -> pd.DataFrame:
    """The official test list (518 videos) is the test split. Every other video
    goes to train or val, split by person so no celebrity is in both.
    A fake whose two people land in different splits is dropped."""
    test_videos = set()
    for line in (celebdf_root / "List_of_testing_videos.txt").read_text().splitlines():
        if line.strip():
            test_videos.add(line.split()[1])
            # e.g. "1 YouTube-real/00170.mp4"

    rows = []
    # Split test first
    for folder in CELEBDF_FOLDERS:
        for path in sorted(require_videos(celebdf_root / folder)):
            rel = f"{folder}/{path.name}"
            rows.append(dict(video_id=f"celebdf/{folder}/{path.stem}", dataset="celebdf",
                             method=folder, label=int(folder == "Celeb-synthesis"),
                             split="test" if rel in test_videos else None,
                             source_ids="_".join(celebdf_identities(path.stem)),
                             path=str(path)))
    df = pd.DataFrame(rows)

    # Then handle train and val
    n_test = (df["split"] == "test").sum()
    if n_test != len(test_videos):
        print(f"[celebdf] warning: {len(test_videos) - n_test} videos in the test list are missing on disk")

    # Pick val people at random (fixed seed) from those in non-test videos.
    # Celebrities and YouTube videos are sampled separately; otherwise the ~300
    # one-video YouTube "people" would take most of the val places.
    rest = df["split"].isna()
    people = sorted(set(df.loc[rest, "source_ids"].str.split("_").explode()))
    rng = random.Random(seed)
    val_people = set()
    for group in ([p for p in people if p.startswith("id")], [p for p in people if p.startswith("yt")]):
        val_people |= set(rng.sample(group, round(val_frac * len(group))))

    def split_for(source_ids: str) -> str | None:
        in_val = {p in val_people for p in source_ids.split("_")}
        if in_val == {True}:
            return "val"
        if in_val == {False}:
            return "train"
        return None  # one person in train, the other in val

    df.loc[rest, "split"] = df.loc[rest, "source_ids"].map(split_for)
    dropped = df["split"].isna()
    n_celebs = sum(p.startswith("id") for p in people)
    n_val_celebs = sum(p.startswith("id") for p in val_people)
    print(f"[celebdf] {n_val_celebs}/{n_celebs} celebrities and "
          f"{len(val_people) - n_val_celebs}/{len(people) - n_celebs} YouTube videos in val; "
          f"dropped {dropped.sum()} fakes that mix train and val people")
    return df[~dropped]


def check_no_identity_leak(df: pd.DataFrame) -> None:
    """Every person (an id in source_ids) must belong to exactly one split."""
    ids = df.assign(identity=df["source_ids"].str.split("_")).explode("identity")
    n_splits = ids.groupby("identity")["split"].nunique()
    leaked = n_splits[n_splits > 1]
    if len(leaked):
        raise AssertionError(f"identities in more than one split: {leaked.index.tolist()[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-root", type=Path, default=Path("~/data").expanduser())
    parser.add_argument("--splits-dir", type=Path, default=REPO_ROOT / "splits" / "ffpp")
    parser.add_argument("--compression", default="c23")
    parser.add_argument("--celebdf-val-frac", type=float, default=0.2,
                        help="share of Celeb-DF people (outside the test list) used for val")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None,
                        help="default: {data_root}/processed/index.csv")
    args = parser.parse_args()

    data_root = args.data_root.expanduser()
    out = args.out or data_root / "processed" / "index.csv"

    ffpp = index_ffpp(data_root / "raw" / "ffpp", args.splits_dir, args.compression)
    check_no_identity_leak(ffpp)
    celebdf = index_celebdf(data_root / "raw" / "celebdf", args.celebdf_val_frac, args.seed)
    # Official test videos share people with the rest, so only train vs val is checked.
    check_no_identity_leak(celebdf[celebdf["split"] != "test"])
    index = pd.concat([ffpp, celebdf], ignore_index=True)
    # video_id, dataset, method, label, split, source_ids, path 

    missing = ~index["path"].map(lambda p: Path(p).exists())
    if missing.any():
        print(f"warning: {missing.sum()} listed files do not exist, e.g. {index.loc[missing, 'path'].iloc[0]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(out, index=False)
    print(f"wrote {len(index)} videos to {out}")
    print(index.groupby(["dataset", "split", "method"]).size().to_string())


if __name__ == "__main__":
    main()
