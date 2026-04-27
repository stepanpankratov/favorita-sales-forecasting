"""CatBoost model: direct single-model approach (no horizon expansion)."""

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from config import CATBOOST_PARAMS, N_FORECAST, SEED
from src.metrics import evaluate
import src.features as F


CAT_FEAT_NAMES = F.CAT_FEATURES
NUM_FEAT_NAMES = F.NUMERIC_FEATURES
# horizon_day removed — one model predicts current-row sales
ALL_FEATURES = NUM_FEAT_NAMES + CAT_FEAT_NAMES + ["onpromotion"]


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN in numeric features and cast cat features to string."""
    df = df.copy()
    for col in NUM_FEAT_NAMES:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
    for col in CAT_FEAT_NAMES:
        if col in df.columns:
            df[col] = df[col].fillna("UNK").astype(str)
    if "onpromotion" in df.columns:
        df["onpromotion"] = df["onpromotion"].fillna(0).astype(float)
    return df


def train_catboost(train_grid: pd.DataFrame, val_grid: pd.DataFrame) -> tuple:
    """
    Train CatBoost regressor directly on grid rows.
    X = feature columns of each row
    y = log1p(sales) of that row  (target column already in grid)

    Returns
    -------
    model       : CatBoostRegressor
    val_metrics : dict
    val_pred_df : pd.DataFrame with store_nbr, item_nbr, date, pred, sales, perishable
    """
    feat_cols = [c for c in ALL_FEATURES if c in train_grid.columns]

    X_train = prepare_features(train_grid)[feat_cols]
    y_train = train_grid["target"].values

    X_val = prepare_features(val_grid)[feat_cols]
    y_val = val_grid["target"].values

    cat_indices = [feat_cols.index(c) for c in CAT_FEAT_NAMES if c in feat_cols]

    model = CatBoostRegressor(**CATBOOST_PARAMS)
    model.fit(
        X_train, y_train,
        cat_features=cat_indices,
        eval_set=(X_val, y_val),
    )

    val_pred_log = model.predict(X_val)
    val_pred = np.expm1(val_pred_log).clip(0)
    y_true = np.expm1(y_val).clip(0)

    perishable = (val_grid["perishable"].fillna(0).values
                  if "perishable" in val_grid.columns
                  else np.zeros(len(val_grid)))
    val_metrics = evaluate(y_true, val_pred, perishable)

    val_pred_df = val_grid[["store_nbr", "item_nbr", "date", "sales"]].copy()
    val_pred_df["pred"] = val_pred
    if "perishable" in val_grid.columns:
        val_pred_df["perishable"] = perishable

    return model, val_metrics, val_pred_df


def predict_catboost(model: CatBoostRegressor, grid: pd.DataFrame,
                     test_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Generate predictions for each test date.

    Uses the last-date snapshot of each (store, item) as the anchor — lag/rolling
    features stay fixed at the anchor; only the date label changes.
    """
    last_date = grid["date"].max()
    anchor = grid[grid["date"] == last_date].copy()
    feat_cols = [c for c in ALL_FEATURES if c in anchor.columns]

    preds = []
    for test_date in sorted(test_dates):
        chunk = anchor.copy()
        chunk["date"] = test_date
        X = prepare_features(chunk)[feat_cols]
        log_pred = model.predict(X)
        pred = np.expm1(log_pred).clip(0)
        chunk["pred"] = pred
        preds.append(chunk[["store_nbr", "item_nbr", "date", "pred"]])

    return pd.concat(preds, ignore_index=True)
