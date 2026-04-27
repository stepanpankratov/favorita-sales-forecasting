"""Statistical baselines on a random sample of store-item series."""

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoETS, AutoTheta

from config import BASELINE_SAMPLE, SEED, N_FORECAST
from src.metrics import evaluate


def sample_series(full_grid: pd.DataFrame, n: int = BASELINE_SAMPLE,
                  rng=None) -> pd.DataFrame:
    """Sample n store-item pairs (approximate baseline on sample)."""
    if rng is None:
        rng = np.random.default_rng(SEED)
    pairs = full_grid[["store_nbr", "item_nbr"]].drop_duplicates()
    n = min(n, len(pairs))
    idx = rng.choice(len(pairs), size=n, replace=False)
    sampled = pairs.iloc[idx]
    return full_grid.merge(sampled, on=["store_nbr", "item_nbr"])


def prepare_sf_df(series_df: pd.DataFrame, val_start: pd.Timestamp) -> tuple:
    """
    Prepare train / actual splits for statsforecast.

    Returns
    -------
    sf_train : pd.DataFrame  in nixtla long format (unique_id, ds, y)
    sf_actual: pd.DataFrame  actual val values (unique_id, ds, y, perishable)
    """
    series_df = series_df.copy()
    series_df["unique_id"] = (
        series_df["store_nbr"].astype(str) + "_" + series_df["item_nbr"].astype(str)
    )

    train_sf = series_df[series_df["date"] < val_start][["unique_id", "date", "sales"]].copy()
    train_sf.columns = ["unique_id", "ds", "y"]
    train_sf["y"] = train_sf["y"].clip(lower=0)

    val_sf = series_df[series_df["date"] >= val_start][
        ["unique_id", "date", "sales", "perishable"]
    ].copy()
    val_sf.columns = ["unique_id", "ds", "y", "perishable"]

    return train_sf, val_sf


def run_baselines(full_grid: pd.DataFrame, val_start: pd.Timestamp,
                  n_sample: int = BASELINE_SAMPLE) -> pd.DataFrame:
    """
    Run Naive, SeasonalNaive(7), AutoTheta(7), AutoETS(7) on sampled series.

    Returns a DataFrame with model names and metrics.
    Note: results are approximate — computed on a random sample of series.
    """
    sample = sample_series(full_grid, n=n_sample)
    train_sf, val_sf = prepare_sf_df(sample, val_start)

    h = N_FORECAST
    models = [
        Naive(),
        SeasonalNaive(season_length=7),
        AutoTheta(season_length=7),
        AutoETS(season_length=7),
    ]

    # Drop series too short for seasonal models (need at least 2×season + 1)
    min_obs = 15
    counts = train_sf.groupby("unique_id")["y"].count()
    keep_ids = counts[counts >= min_obs].index
    train_sf = train_sf[train_sf["unique_id"].isin(keep_ids)]
    val_sf = val_sf[val_sf["unique_id"].isin(keep_ids)]

    sf = StatsForecast(models=models, freq="D", n_jobs=-1)
    sf.fit(train_sf)
    forecasts = sf.predict(h=h)

    # Merge with actuals
    merged = val_sf.merge(forecasts, on=["unique_id", "ds"], how="inner")

    perishable = merged["perishable"].fillna(0).values

    results = []
    for model_name in ["Naive", "SeasonalNaive", "AutoTheta", "AutoETS"]:
        if model_name not in merged.columns:
            continue
        y_pred = merged[model_name].clip(lower=0).values
        y_true = merged["y"].clip(lower=0).values
        metrics = evaluate(y_true, y_pred, perishable)
        results.append({"model": model_name, **metrics})

    return pd.DataFrame(results)
