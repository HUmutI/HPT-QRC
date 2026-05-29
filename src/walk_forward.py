"""
walk_forward.py
===============
Rolling walk-forward splits for time-series benchmarks.

Per `PROTOCOL.md` §3:
  - For monthly RV (817 months ~ 1950-2017): use a 60-month train, 6-month val,
    6-month test, slide forward by 6 months. Yields >40 folds; we cap or
    subsample for tractability.
  - For daily VIX (~6,000 daily samples): use a 5-year (≈1260 day) train,
    6-month (~126 day) val, 6-month test, slide forward by 6 months.

The iterator is data-frequency-agnostic: pass window sizes in *index steps*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import pandas as pd


@dataclass
class Fold:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def slice_df(self, df: pd.DataFrame):
        """Return (train, val, test) DataFrames sliced from `df`."""
        tr = df.loc[(df.index >= self.train_start) & (df.index <= self.train_end)]
        va = df.loc[(df.index >= self.val_start) & (df.index <= self.val_end)]
        te = df.loc[(df.index >= self.test_start) & (df.index <= self.test_end)]
        return tr, va, te


class WalkForwardSplit:
    """
    Walk-forward iterator over a DatetimeIndex-indexed DataFrame.

    Parameters
    ----------
    train_steps : int   # length of train block (index steps, not days)
    val_steps   : int   # length of val block
    test_steps  : int   # length of test block
    step_steps  : int   # how far to slide between folds
    max_folds   : optional int   # cap on number of folds
    """

    def __init__(
        self,
        train_steps: int,
        val_steps: int,
        test_steps: int,
        step_steps: int,
        max_folds: int | None = None,
    ):
        self.train_steps = train_steps
        self.val_steps = val_steps
        self.test_steps = test_steps
        self.step_steps = step_steps
        self.max_folds = max_folds

    def split(self, df: pd.DataFrame) -> Iterator[Fold]:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("WalkForwardSplit requires a DatetimeIndex.")
        n = len(df)
        block = self.train_steps + self.val_steps + self.test_steps
        if n < block:
            raise ValueError(f"Not enough rows ({n}) for one fold ({block}).")
        fold_id = 0
        start = 0
        while start + block <= n:
            tr_lo = start
            tr_hi = start + self.train_steps - 1
            va_lo = tr_hi + 1
            va_hi = va_lo + self.val_steps - 1
            te_lo = va_hi + 1
            te_hi = te_lo + self.test_steps - 1
            yield Fold(
                fold_id=fold_id,
                train_start=df.index[tr_lo],
                train_end=df.index[tr_hi],
                val_start=df.index[va_lo],
                val_end=df.index[va_hi],
                test_start=df.index[te_lo],
                test_end=df.index[te_hi],
            )
            fold_id += 1
            if self.max_folds is not None and fold_id >= self.max_folds:
                break
            start += self.step_steps

    def fold_count(self, df: pd.DataFrame) -> int:
        return sum(1 for _ in self.split(df))


# ---------------------------------------------------------------------------
# Recommended defaults
# ---------------------------------------------------------------------------
def sp500_monthly_split(max_folds: int | None = 8) -> WalkForwardSplit:
    """Train 240 months (20yr), val 24, test 24, slide 24. ~8 folds across 1950-2017."""
    return WalkForwardSplit(train_steps=240, val_steps=24, test_steps=24,
                            step_steps=24, max_folds=max_folds)


def vix_daily_split(max_folds: int | None = 8) -> WalkForwardSplit:
    """Train ~5yr (1260), val ~6mo (126), test ~6mo (126), slide 126. 8 folds."""
    return WalkForwardSplit(train_steps=1260, val_steps=126, test_steps=126,
                            step_steps=126, max_folds=max_folds)


if __name__ == "__main__":
    # Smoke test on synthetic data
    import numpy as np
    idx = pd.date_range("2010-01-01", periods=300, freq="ME")
    df = pd.DataFrame({"y": np.arange(300, dtype=float)}, index=idx)
    splitter = sp500_monthly_split(max_folds=5)
    for f in splitter.split(df):
        tr, va, te = f.slice_df(df)
        print(f"fold {f.fold_id}: train {len(tr):3d} val {len(va):3d} test {len(te):3d}  "
              f"[{f.train_start.date()} .. {f.test_end.date()}]")
