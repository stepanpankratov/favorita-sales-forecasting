"""Preprocessing: time filtering, full grid construction, target encoding."""

import numpy as np
import pandas as pd
from config import N_DAYS_HISTORY, N_VAL_DAYS, GRID_MODE


def filter_and_split(train: pd.DataFrame):
    """
    Step 1: Keep last N_DAYS_HISTORY days from train.
    Step 2: Split into train / val by date.

    Returns
    -------
    train_part : pd.DataFrame  — rows with date < val_start
    val_part   : pd.DataFrame  — rows with date >= val_start
    max_date   : pd.Timestamp
    val_start  : pd.Timestamp
    """
    max_date = train["date"].max()
    window_start = max_date - pd.Timedelta(days=N_DAYS_HISTORY - 1)
    train = train[train["date"] >= window_start].copy()

    val_start = max_date - pd.Timedelta(days=N_VAL_DAYS - 1)
    train_part = train[train["date"] < val_start].copy()
    val_part = train[train["date"] >= val_start].copy()

    return train_part, val_part, max_date, val_start


def _meta_cols(train_raw: pd.DataFrame) -> list[str]:
    return [c for c in ["city", "state", "type", "cluster", "family", "class", "perishable"]
            if c in train_raw.columns]


def _build_sparse_grid(train_raw: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Sparse mode: use only existing (date, store, item) records from train.
    No cross-join — avoids memory explosion for large store-item spaces.
    """
    meta_cols = _meta_cols(train_raw)
    keep_cols = (["date", "store_nbr", "item_nbr", "sales", "onpromotion", "dcoilwtico"]
                 + meta_cols)
    keep_cols = [c for c in keep_cols if c in train_raw.columns]

    dates_set = set(dates)
    grid = (train_raw[keep_cols]
            .drop_duplicates(["date", "store_nbr", "item_nbr"])
            .copy())
    grid = grid[grid["date"].isin(dates_set)]
    grid["sales"] = grid["sales"].fillna(0).clip(lower=0)
    grid["onpromotion"] = grid["onpromotion"].fillna(0).astype(int)
    return grid.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)


def build_full_grid(train_raw: pd.DataFrame, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Build date × store_nbr × item_nbr grid for *existing* store-item pairs only.
    Missing rows get sales=0, onpromotion=0.

    GRID_MODE='sparse': just use actual records (no cross-join).
    GRID_MODE='full':   cross-join all dates × existing pairs (memory-intensive).
    """
    if GRID_MODE == "sparse":
        return _build_sparse_grid(train_raw, dates)

    # --- full cross-join mode ---
    existing_pairs = train_raw[["store_nbr", "item_nbr"]].drop_duplicates()
    dates_df = pd.DataFrame({"date": dates})
    dates_df["key"] = 1
    existing_pairs = existing_pairs.copy()
    existing_pairs["key"] = 1

    grid = dates_df.merge(existing_pairs, on="key").drop(columns="key")

    meta_cols = _meta_cols(train_raw)
    keep_cols = (["date", "store_nbr", "item_nbr", "sales", "onpromotion", "dcoilwtico"]
                 + meta_cols)
    keep_cols = [c for c in keep_cols if c in train_raw.columns]
    grid = grid.merge(train_raw[keep_cols].drop_duplicates(["date", "store_nbr", "item_nbr"]),
                      on=["date", "store_nbr", "item_nbr"], how="left")

    grid["sales"] = grid["sales"].fillna(0).clip(lower=0)
    grid["onpromotion"] = grid["onpromotion"].fillna(0).astype(int)

    item_meta = (train_raw[["item_nbr"] + [c for c in ["family", "class", "perishable"]
                                            if c in train_raw.columns]]
                 .drop_duplicates("item_nbr"))
    store_meta = (train_raw[["store_nbr"] + [c for c in ["city", "state", "type", "cluster"]
                                              if c in train_raw.columns]]
                  .drop_duplicates("store_nbr"))

    for col in [c for c in ["family", "class", "perishable"] if c in item_meta.columns]:
        if grid[col].isna().any():
            grid = grid.drop(columns=[col]).merge(item_meta[["item_nbr", col]], on="item_nbr", how="left")

    for col in [c for c in ["city", "state", "type", "cluster"] if c in store_meta.columns]:
        if grid[col].isna().any():
            grid = grid.drop(columns=[col]).merge(store_meta[["store_nbr", col]], on="store_nbr", how="left")

    return grid.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Add log1p target column. Keep original sales intact."""
    df = df.copy()
    df["target"] = np.log1p(df["sales"].clip(lower=0))
    return df


def preprocess(train_raw: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Returns
    -------
    train_grid : pd.DataFrame  — full grid, train part, with target
    val_grid   : pd.DataFrame  — full grid, val part, with target
    full_grid  : pd.DataFrame  — full windowed grid (train+val), with target
    val_start  : pd.Timestamp
    max_date   : pd.Timestamp
    """
    train_part, val_part, max_date, val_start = filter_and_split(train_raw)

    window_start = max_date - pd.Timedelta(days=N_DAYS_HISTORY - 1)
    all_dates = pd.date_range(window_start, max_date, freq="D")

    full_grid = build_full_grid(train_raw[train_raw["date"] >= window_start], all_dates)
    full_grid = add_target(full_grid)

    train_grid = full_grid[full_grid["date"] < val_start].copy()
    val_grid = full_grid[full_grid["date"] >= val_start].copy()

    return train_grid, val_grid, full_grid, val_start, max_date
