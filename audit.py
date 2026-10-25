from __future__ import annotations

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from pandas.api.types import is_numeric_dtype, is_bool_dtype


def _is_missing(value):
    if pd.isna(value):
        return True

    if isinstance(value, str):
        return value.strip() == ""

    return False


def audit(df: pd.DataFrame, *, target: str, id_col: str, time_col: str) -> dict:
    for col in (target, id_col, time_col):
        if col not in df.columns:
            raise KeyError(col)

    n_rows = len(df)

    missing = {
        col: float(df[col].apply(_is_missing).mean())
        for col in df.columns
    }

    # duplicate ids
    id_mask = ~df[id_col].apply(_is_missing)

    duplicate_ids = sorted(
        df.loc[id_mask, id_col]
        .loc[lambda s: s.duplicated(keep=False)]
        .unique()
        .tolist()
    )

    # duplicate rows
    duplicate_rows = int(df.duplicated().sum())

    # constant columns
    constant_columns = []

    for col in df.columns:
        if col == target:
            continue

        non_missing = df.loc[~df[col].apply(_is_missing), col]

        if non_missing.nunique(dropna=True) <= 1:
            constant_columns.append(col)

    constant_columns = sorted(constant_columns)

    # unlabelled rows
    label_missing = df[target].apply(_is_missing)

    unlabelled_rows = int(label_missing.sum())

    labelled = df.loc[~label_missing]

    if len(labelled) == 0:
        positive_rate = 0.0
    else:
        positive_rate = float((labelled[target] == 1).mean())

    # timestamp parsing
    time_missing = df[time_col].apply(_is_missing)

    raw_times = df.loc[~time_missing, time_col]

    parsed = pd.to_datetime(raw_times, errors="coerce")

    bad_timestamps = int(parsed.isna().sum())

    valid_times = parsed.dropna()

    if len(valid_times) == 0:
        time_range = [None, None]
    else:
        time_range = [
            valid_times.min().isoformat(),
            valid_times.max().isoformat(),
        ]

    # leakage detection
    leakage_suspects = set()

    if len(labelled) >= 20 and labelled[target].nunique() == 2:
        y = labelled[target]

        for col in df.columns:
            if col in {target, id_col, time_col}:
                continue

            series = labelled[col]

            # skip columns with only one value
            values_with_missing = series.apply(
                lambda x: "__MISSING__" if _is_missing(x) else x
            )

            if values_with_missing.nunique(dropna=False) <= 1:
                continue

            # (b) missingness leakage
            try:
                missing_indicator = series.apply(
                    lambda x: 1 if _is_missing(x) else 0
                )

                if missing_indicator.nunique() > 1:
                    auc = roc_auc_score(y, missing_indicator)

                    if auc >= 0.98 or auc <= 0.02:
                        leakage_suspects.add(col)
            except ValueError:
                pass

            # (a) value leakage
            if is_numeric_dtype(series) or is_bool_dtype(series):
                mask = ~series.apply(_is_missing)

                if mask.sum() < 20:
                    continue

                y_col = y[mask]
                x_col = series[mask]

                if y_col.nunique() < 2:
                    continue

                if x_col.nunique(dropna=True) <= 1:
                    continue

                try:
                    auc = roc_auc_score(y_col, x_col)

                    if auc >= 0.98 or auc <= 0.02:
                        leakage_suspects.add(col)

                except ValueError:
                    pass

    leakage_suspects = sorted(leakage_suspects)

    return {
        "n_rows": int(n_rows),
        "missing": missing,
        "duplicate_ids": duplicate_ids,
        "duplicate_rows": duplicate_rows,
        "constant_columns": constant_columns,
        "unlabelled_rows": unlabelled_rows,
        "positive_rate": positive_rate,
        "bad_timestamps": bad_timestamps,
        "time_range": time_range,
        "leakage_suspects": leakage_suspects,
    }