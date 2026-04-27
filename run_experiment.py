"""
Main experiment runner.

Usage:
    python run_experiment.py --stage all
    python run_experiment.py --stage baselines
    python run_experiment.py --stage catboost
    python run_experiment.py --stage nn
    python run_experiment.py --stage ensemble
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Unbuffered stdout so progress is visible in pipes / background tasks
sys.stdout.reconfigure(line_buffering=True)

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_DIR, RESULTS_DIR, SEED, NN_SAMPLE_PAIRS
from src.data_loader import load_and_merge
from src.preprocessing import preprocess
from src.features import build_features
from src.metrics import evaluate


def save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"Saved: {path}")


def _sample_pairs(df: pd.DataFrame, n_pairs: int, seed: int = SEED) -> pd.DataFrame:
    """Subsample to n_pairs random (store_nbr, item_nbr) pairs."""
    pairs = df[["store_nbr", "item_nbr"]].drop_duplicates()
    if len(pairs) <= n_pairs:
        return df
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(pairs), size=n_pairs, replace=False)
    keep = pairs.iloc[idx]
    return df.merge(keep, on=["store_nbr", "item_nbr"])


def stage_baselines(full_grid, val_start):
    from src.baselines import run_baselines

    print("\n=== Baselines (approximate, sampled) ===")
    results = run_baselines(full_grid, val_start)
    print(results.to_string(index=False))

    path = os.path.join(RESULTS_DIR, "baselines.csv")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(path, index=False)
    print(f"Saved: {path}")
    return results


def stage_catboost(train_grid_feat, val_grid_feat, full_grid_feat):
    from src.catboost_model import train_catboost, predict_catboost

    print("\n=== CatBoost ===")
    model, val_metrics, val_pred_df = train_catboost(train_grid_feat, val_grid_feat)
    print(f"Val NWRMSLE: {val_metrics['nwrmsle']:.4f}  WMAPE: {val_metrics.get('wmape', float('nan')):.4f}")

    path = os.path.join(RESULTS_DIR, "catboost_val_metrics.json")
    save_json(val_metrics, path)

    val_pred_df.to_csv(os.path.join(RESULTS_DIR, "catboost_val_preds.csv"), index=False)
    print(f"Saved val preds: {os.path.join(RESULTS_DIR, 'catboost_val_preds.csv')}")

    # Save feature importance for notebook
    fi = pd.DataFrame({
        "feature": model.feature_names_,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)
    fi.to_csv(os.path.join(RESULTS_DIR, "catboost_feature_importance.csv"), index=False)
    print(f"Saved feature importance: {os.path.join(RESULTS_DIR, 'catboost_feature_importance.csv')}")

    return model, val_metrics, val_pred_df


def stage_nn(train_grid_feat, val_grid_feat):
    from src.nn_model import train_nn

    # MIMO needs the full grid (train+val) so targets can span the boundary
    full_for_nn = pd.concat([train_grid_feat, val_grid_feat], ignore_index=True)
    val_start = val_grid_feat["date"].min()

    # Subsample pairs for NN to keep memory manageable
    if NN_SAMPLE_PAIRS is not None:
        n_before = len(full_for_nn)
        full_for_nn = _sample_pairs(full_for_nn, NN_SAMPLE_PAIRS)
        print(f"  NN subsample: {n_before:,} → {len(full_for_nn):,} rows "
              f"({NN_SAMPLE_PAIRS} pairs)")

    print("\n=== Neural Network (MIMO MLP) ===")
    model, cat_encoder, cat_cols, num_cols, val_metrics, val_pred_df = train_nn(
        full_for_nn, val_start
    )
    print(f"Val NWRMSLE: {val_metrics['nwrmsle']:.4f}")

    path = os.path.join(RESULTS_DIR, "nn_val_metrics.json")
    save_json(val_metrics, path)

    return model, cat_encoder, cat_cols, num_cols, val_metrics, val_pred_df


def stage_ensemble(cb_val_pred_df, nn_val_pred_df, val_grid_feat,
                   cb_model, full_grid_feat,
                   nn_model, cat_encoder, cat_cols, num_cols):
    from src.ensemble import find_best_alpha, ensemble_predict
    from src.catboost_model import predict_catboost
    from src.nn_model import predict_nn

    print("\n=== Ensemble ===")
    best_alpha = find_best_alpha(cb_val_pred_df, nn_val_pred_df, val_grid_feat)

    # Generate test-time predictions (using full grid as context)
    test_dates = pd.date_range(
        full_grid_feat["date"].max() + pd.Timedelta(days=1), periods=16, freq="D"
    )
    cb_test_pred = predict_catboost(cb_model, full_grid_feat, test_dates)
    nn_test_pred = predict_nn(nn_model, cat_encoder, cat_cols, num_cols, full_grid_feat)

    ens_pred = ensemble_predict(cb_test_pred, nn_test_pred, alpha=best_alpha)
    path = os.path.join(RESULTS_DIR, "ensemble_test_preds.csv")
    ens_pred.to_csv(path, index=False)
    print(f"Saved ensemble predictions: {path}")

    save_json({"best_alpha": best_alpha}, os.path.join(RESULTS_DIR, "ensemble_alpha.json"))
    return ens_pred, best_alpha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "baselines", "catboost", "nn", "ensemble"])
    args = parser.parse_args()

    needs_full_grid_feat = args.stage in ("all", "baselines", "catboost", "ensemble")

    print("Loading data...")
    train_raw, test_raw, oil, transactions, holidays = load_and_merge(DATA_DIR)

    print("Preprocessing...")
    train_grid, val_grid, full_grid, val_start, max_date = preprocess(train_raw)
    print(f"  Train: {len(train_grid):,} rows  |  Val: {len(val_grid):,} rows")
    print(f"  Val start: {val_start.date()}  |  Max train date: {max_date.date()}")

    print("Building features (train)...")
    train_grid_feat = build_features(train_grid, transactions, holidays, oil)
    print("Building features (val)...")
    val_grid_feat = build_features(val_grid, transactions, holidays, oil)

    full_grid_feat = None
    if needs_full_grid_feat:
        print("Building features (full grid)...")
        full_grid_feat = build_features(full_grid, transactions, holidays, oil)

    print(f"  Feature columns: {train_grid_feat.shape[1]}")

    results_summary = {}

    if args.stage in ("all", "baselines"):
        bl_results = stage_baselines(full_grid_feat, val_start)
        results_summary["baselines"] = bl_results.to_dict(orient="records")

    cb_model = cb_val_pred_df = cb_val_metrics = None
    nn_model = nn_cat_encoder = nn_cat_cols = nn_num_cols = nn_val_pred_df = None

    if args.stage in ("all", "catboost"):
        cb_model, cb_val_metrics, cb_val_pred_df = stage_catboost(
            train_grid_feat, val_grid_feat, full_grid_feat
        )
        results_summary["catboost"] = cb_val_metrics

    if args.stage in ("all", "nn"):
        nn_model, nn_cat_encoder, nn_cat_cols, nn_num_cols, nn_val_metrics, nn_val_pred_df = stage_nn(
            train_grid_feat, val_grid_feat
        )
        results_summary["nn"] = nn_val_metrics

    if args.stage in ("all", "ensemble"):
        if cb_model is None or nn_model is None:
            print("ERROR: ensemble stage requires both catboost and nn to be run first.")
            sys.exit(1)
        ens_pred, best_alpha = stage_ensemble(
            cb_val_pred_df, nn_val_pred_df, val_grid_feat,
            cb_model, full_grid_feat,
            nn_model, nn_cat_encoder, nn_cat_cols, nn_num_cols,
        )
        results_summary["ensemble_alpha"] = best_alpha

    save_json(results_summary, os.path.join(RESULTS_DIR, "results_summary.json"))
    print("\nDone.")


if __name__ == "__main__":
    main()
