"""Ensemble: grid search alpha on val NWRMSLE, then blend CatBoost + NN."""

import numpy as np
import pandas as pd

from src.metrics import evaluate, make_weights


def blend(cb_pred: np.ndarray, nn_pred: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * cb_pred + (1 - alpha) * nn_pred


def find_best_alpha(cb_val_df: pd.DataFrame, nn_val_df: pd.DataFrame,
                    val_grid: pd.DataFrame) -> float:
    """
    Grid search alpha ∈ [0.1, 0.9] step 0.1 on val NWRMSLE.

    Parameters
    ----------
    cb_val_df : DataFrame with columns [store_nbr, item_nbr, date, pred]  — CatBoost val preds
    nn_val_df : DataFrame with columns [store_nbr, item_nbr, date, pred]  — NN val preds
    val_grid  : full val grid with sales and perishable columns
    """
    # Merge both sets of predictions with actuals
    actuals = val_grid[["store_nbr", "item_nbr", "date", "sales", "perishable"]].copy()
    actuals["perishable"] = actuals["perishable"].fillna(0)

    cb = cb_val_df[["store_nbr", "item_nbr", "date", "pred"]].rename(columns={"pred": "cb_pred"})
    nn = nn_val_df[["store_nbr", "item_nbr", "date", "pred"]].rename(columns={"pred": "nn_pred"})

    df = actuals.merge(cb, on=["store_nbr", "item_nbr", "date"], how="inner")
    df = df.merge(nn, on=["store_nbr", "item_nbr", "date"], how="inner")

    y_true = df["sales"].clip(lower=0).values
    cb_p = df["cb_pred"].clip(lower=0).values
    nn_p = df["nn_pred"].clip(lower=0).values
    perishable = df["perishable"].values

    best_alpha = 0.5
    best_score = float("inf")

    for alpha in np.arange(0.1, 1.0, 0.1):
        blended = blend(cb_p, nn_p, alpha)
        score = evaluate(y_true, blended, perishable)["nwrmsle"]
        print(f"  alpha={alpha:.1f}  nwrmsle={score:.4f}")
        if score < best_score:
            best_score = score
            best_alpha = round(alpha, 1)

    print(f"Best alpha={best_alpha}  nwrmsle={best_score:.4f}")
    return best_alpha


def ensemble_predict(cb_pred_df: pd.DataFrame, nn_pred_df: pd.DataFrame,
                     alpha: float) -> pd.DataFrame:
    """
    Blend CatBoost and NN predictions using given alpha.

    Parameters
    ----------
    cb_pred_df : DataFrame [store_nbr, item_nbr, date, pred]
    nn_pred_df : DataFrame [store_nbr, item_nbr, date, pred]
    alpha      : weight for CatBoost

    Returns
    -------
    DataFrame [store_nbr, item_nbr, date, pred]
    """
    merged = cb_pred_df[["store_nbr", "item_nbr", "date", "pred"]].rename(
        columns={"pred": "cb_pred"}
    ).merge(
        nn_pred_df[["store_nbr", "item_nbr", "date", "pred"]].rename(columns={"pred": "nn_pred"}),
        on=["store_nbr", "item_nbr", "date"],
        how="outer",
    )

    # Fill missing predictions with the other model's value
    merged["cb_pred"] = merged["cb_pred"].fillna(merged["nn_pred"])
    merged["nn_pred"] = merged["nn_pred"].fillna(merged["cb_pred"])

    merged["pred"] = blend(merged["cb_pred"].values, merged["nn_pred"].values, alpha)
    return merged[["store_nbr", "item_nbr", "date", "pred"]]
