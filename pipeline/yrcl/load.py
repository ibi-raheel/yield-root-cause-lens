"""Load the UCI SECOM files (590 measurements + pass/fail label + timestamp per unit)."""
from pathlib import Path

import numpy as np
import pandas as pd


def load_secom(data_dir: Path) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return (X, fail, timestamp), sorted by timestamp.

    fail is 1 for a failing unit (label 1 in the source) and 0 for a pass (-1).
    """
    data_dir = Path(data_dir)
    X = pd.read_csv(data_dir / "secom.data", sep=r"\s+", header=None, na_values="NaN")
    X.columns = [f"S{i:03d}" for i in range(X.shape[1])]
    lab = pd.read_csv(data_dir / "secom_labels.data", sep=" ", header=None, names=["y", "ts"], quotechar='"')
    ts = pd.to_datetime(lab["ts"], format="%d/%m/%Y %H:%M:%S")
    fail = (lab["y"] == 1).astype(int)
    order = np.argsort(ts.values, kind="stable")
    X = X.iloc[order].reset_index(drop=True)
    fail = fail.iloc[order].reset_index(drop=True).rename("fail")
    ts = ts.iloc[order].reset_index(drop=True).rename("ts")
    return X, fail, ts
