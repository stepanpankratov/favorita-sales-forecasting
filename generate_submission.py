"""Generate Kaggle submission using CatBoost (best single model)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd

from config import DATA_DIR, RESULTS_DIR, CATBOOST_PARAMS, SEED
from src.data_loader import load_and_merge
from src.preprocessing import preprocess
from src.features import build_features
from src.catboost_model import prepare_features, ALL_FEATURES, CAT_FEAT_NAMES

from catboost import CatBoostRegressor


def main():
    print("Loading data...")
    train_raw, test_raw, oil, transactions, holidays = load_and_merge(DATA_DIR)

    print("Preprocessing...")
    train_grid, val_grid, full_grid, val_start, max_date = preprocess(train_raw)
    print(f"  Full grid: {len(full_grid):,} rows | max_date: {max_date.date()}")

    # For final submission: train on FULL grid (train + val)
    print("Building features on full grid...")
    full_grid_feat = build_features(full_grid, transactions, holidays, oil)

    feat_cols = [c for c in ALL_FEATURES if c in full_grid_feat.columns]
    X_train = prepare_features(full_grid_feat)[feat_cols]
    y_train = full_grid_feat["target"].values
    cat_indices = [feat_cols.index(c) for c in CAT_FEAT_NAMES if c in feat_cols]

    # Train with fixed iterations (model early-stopped at ~480 on val)
    params = {**CATBOOST_PARAMS}
    params.pop("early_stopping_rounds", None)
    params["iterations"] = 500

    print("Training CatBoost on full data (500 iterations)...")
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, cat_features=cat_indices)

    # --- Predict on test ---
    print("Generating predictions for test...")
    last_date = full_grid_feat["date"].max()
    anchor = full_grid_feat[full_grid_feat["date"] == last_date].copy()

    test_dates = sorted(test_raw["date"].unique())
    preds_list = []

    for test_date in test_dates:
        # Get test onpromotion for this date
        test_day = test_raw[test_raw["date"] == test_date][
            ["store_nbr", "item_nbr", "onpromotion"]
        ].copy()
        test_day["onpromotion"] = test_day["onpromotion"].fillna(0)

        # Merge anchor features with test onpromotion
        chunk = anchor.drop(columns=["onpromotion"], errors="ignore").merge(
            test_day, on=["store_nbr", "item_nbr"], how="inner"
        )
        chunk["date"] = test_date

        X = prepare_features(chunk)[feat_cols]
        log_pred = model.predict(X)
        pred = np.expm1(log_pred).clip(0)

        chunk["pred"] = pred
        preds_list.append(chunk[["store_nbr", "item_nbr", "date", "pred"]])

    predictions = pd.concat(preds_list, ignore_index=True)

    # Merge with test.csv to get id column
    submission = test_raw[["id", "date", "store_nbr", "item_nbr"]].merge(
        predictions, on=["date", "store_nbr", "item_nbr"], how="left"
    )
    submission["unit_sales"] = submission["pred"].fillna(0).clip(lower=0)
    submission = submission[["id", "unit_sales"]]

    # Save
    path = os.path.join(RESULTS_DIR, "submission.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    submission.to_csv(path, index=False)

    n_missing = (submission["unit_sales"] == 0).sum()
    print(f"\nSubmission saved: {path}")
    print(f"  Rows:            {len(submission):,}")
    print(f"  Mean prediction: {submission['unit_sales'].mean():.2f}")
    print(f"  Zeros (missing): {n_missing:,} ({100*n_missing/len(submission):.1f}%)")


if __name__ == "__main__":
    main()
