from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd


def load_pairs_csv(pairs_path: Path) -> pd.DataFrame:
    return pd.read_csv(pairs_path)


def iter_matcher_pairs(df: pd.DataFrame) -> Iterable[Tuple[str, str]]:
    for _, row in df.iterrows():
        yield row["i"], row["j"]
