"""NWRMSLE and WMAPE metrics for Favorita evaluation."""

import numpy as np
import pandas as pd


def nwrmsle(y_true: np.ndarray, y_pred: np.ndarray,
            weights: np.ndarray) -> float:
    """
    Normalized Weighted Root Mean Squared Logarithmic Error.

    NWRMSLE = sqrt( sum(w * (log(y_pred+1) - log(y_true+1))^2) / sum(w) )

    Perishable items have weight 1.25, non-perishable 1.0.
    """
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    return float(np.sqrt(np.sum(weights * log_diff ** 2) / np.sum(weights)))


def wmape(y_true: np.ndarray, y_pred: np.ndarray,
          weights: np.ndarray) -> float:
    """Weighted Mean Absolute Percentage Error."""
    y_pred = np.clip(y_pred, 0, None)
    y_true = np.clip(y_true, 0, None)
    denom = np.sum(weights * y_true)
    if denom == 0:
        return float("nan")
    return float(np.sum(weights * np.abs(y_true - y_pred)) / denom)


def make_weights(perishable: np.ndarray) -> np.ndarray:
    """Return weight array: 1.25 for perishable, 1.0 otherwise."""
    return np.where(perishable == 1, 1.25, 1.0)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             perishable: np.ndarray) -> dict:
    """Return dict with nwrmsle and wmape."""
    w = make_weights(perishable)
    return {
        "nwrmsle": nwrmsle(y_true, y_pred, w),
        "wmape": wmape(y_true, y_pred, w),
    }
