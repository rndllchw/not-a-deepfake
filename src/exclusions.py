"""
Videos left out of every metric: DFDC fakes that change only the audio
(flagged by src.preprocessing.dfdc_audio_only). Our models see frames only, so
these fakes cannot be detected from what the models are given.

They are still scored and kept in every prediction file; they are dropped only when
metrics are computed. Each dataset that loses videos also gets a
<dataset>_unfiltered entry with them, so the numbers before the filter stay
reportable (e.g. dfdc and dfdc_unfiltered in the comparison tables).
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd

AUDIO_ONLY = Path("processed") / "dfdc_audio_only.csv"


def excluded_videos(data_root: Path) -> set[str]:
    """video_ids to leave out of the metrics. Empty, with a note, if the audio-only list has not been made."""
    path = Path(data_root).expanduser() / AUDIO_ONLY
    if not path.exists():
        print(f"note: no {path}, so no DFDC fakes are excluded (run src.preprocessing.dfdc_audio_only)")
        return set()
    flags = pd.read_csv(path)
    return set(flags.loc[flags["audio_only"].astype(bool), "video_id"])


def drop_excluded(table: pd.DataFrame, excluded: set[str]) -> pd.DataFrame:
    """`table` (with a video_id column) without the excluded videos."""
    return table[~table["video_id"].isin(excluded)]


def metric_parts(table: pd.DataFrame, excluded: set[str]) -> Iterator[tuple[str, pd.DataFrame]]:
    """(name, rows) per dataset for the metrics: each dataset without the excluded videos,
    and <dataset>_unfiltered with all of its rows for every dataset that had some excluded."""
    for dataset, part in table.groupby("dataset"):
        kept = drop_excluded(part, excluded)
        yield dataset, kept
        if len(kept) < len(part):
            yield f"{dataset}_unfiltered", part
