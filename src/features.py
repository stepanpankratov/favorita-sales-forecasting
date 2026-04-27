"""Feature engineering — strict no-leakage: all rolling/lag use shift(1) first."""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _days_since_last_nonzero(sales: pd.Series) -> pd.Series:
    """Days since last nonzero sales (only past data via shift(1)). Vectorized."""
    shifted = sales.shift(1)
    is_nonzero = (shifted > 0).fillna(False)
    # Each nonzero increments the group counter; cumcount within each group
    # gives the number of consecutive zeros since that nonzero.
    groups = is_nonzero.cumsum()
    result = groups.groupby(groups).cumcount().astype(float)
    result[shifted.isna()] = np.nan
    return result


def _promo_streak(promo: pd.Series) -> pd.Series:
    """Consecutive days on promotion (past-only, uses shift(1)). Vectorized."""
    shifted = promo.shift(1).fillna(0).astype(int)
    # Every time promo resets to 0, start a new group; cumsum within group = streak
    reset = (shifted == 0).astype(int)
    group = reset.cumsum()
    return shifted.groupby(group).cumsum().astype(float)


def _grp_roll(series: pd.Series, groups: list, window: int, func: str,
              min_periods: int = 1, n_group_levels: int = 2) -> pd.Series:
    """
    Apply grouped rolling operation and return a Series aligned to original index.

    Parameters
    ----------
    series        : already-shifted Series (same index as the parent DataFrame)
    groups        : list of Series used as groupby keys (e.g. [df["store_nbr"], df["item_nbr"]])
    window        : rolling window size
    func          : "mean" | "std" | "median" | "sum"
    n_group_levels: number of group-key levels to drop from the result MultiIndex
    """
    grp = series.groupby(groups).rolling(window, min_periods=min_periods)
    if func == "mean":
        result = grp.mean()
    elif func == "std":
        result = grp.std()
    elif func == "median":
        result = grp.median()
    elif func == "sum":
        result = grp.sum()
    else:
        raise ValueError(f"Unknown func: {func}")
    return result.reset_index(level=list(range(n_group_levels)), drop=True)


# ---------------------------------------------------------------------------
# Holiday features
# ---------------------------------------------------------------------------

def build_holiday_features(df: pd.DataFrame, holidays: pd.DataFrame) -> pd.DataFrame:
    """
    Add is_holiday_local and holiday_type columns.

    is_holiday_local: bool — local holiday (not transferred)
    holiday_type: str — National / Regional / Local / None
    """
    hol = holidays[holidays["transferred"] == False].copy()

    national = hol[hol["locale"] == "National"][["date", "type"]].copy()
    national.columns = ["date", "holiday_type"]
    national = national.drop_duplicates("date")

    regional = hol[hol["locale"] == "Regional"][["date", "locale_name", "type"]].copy()
    regional.columns = ["date", "state", "holiday_type_r"]

    local_hol = hol[hol["locale"] == "Local"][["date", "locale_name", "type"]].copy()
    local_hol.columns = ["date", "city", "holiday_type_l"]
    local_hol["is_holiday_local"] = True

    df = df.merge(national, on="date", how="left")
    df["holiday_type"] = df["holiday_type"].fillna("None")

    if "state" in df.columns:
        df = df.merge(regional, on=["date", "state"], how="left")
        mask = (df["holiday_type"] == "None") & df["holiday_type_r"].notna()
        df.loc[mask, "holiday_type"] = df.loc[mask, "holiday_type_r"]
        df.drop(columns=["holiday_type_r"], inplace=True)

    if "city" in df.columns:
        df = df.merge(local_hol, on=["date", "city"], how="left")
        mask = (df["holiday_type"] == "None") & df["holiday_type_l"].notna()
        df.loc[mask, "holiday_type"] = df.loc[mask, "holiday_type_l"]
        df["is_holiday_local"] = df["is_holiday_local"].fillna(False).astype(bool)
        df.drop(columns=["holiday_type_l"], inplace=True)
    else:
        df["is_holiday_local"] = False

    return df


# ---------------------------------------------------------------------------
# Transaction features (built separately, merged by date+store)
# ---------------------------------------------------------------------------

def build_transaction_features(df: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    """Add transactions_lag1 and transactions_mean7 — no leakage."""
    tx = transactions[["date", "store_nbr", "transactions"]].copy()
    tx = tx.sort_values(["store_nbr", "date"]).reset_index(drop=True)

    tx["transactions_lag1"] = tx.groupby("store_nbr")["transactions"].shift(1)

    # Rolling mean on the already-shifted values — no lambda
    tx_s1 = tx.groupby("store_nbr")["transactions"].shift(1)
    tx["transactions_mean7"] = (
        tx_s1.groupby(tx["store_nbr"])
        .rolling(7, min_periods=1).mean()
        .reset_index(level=0, drop=True)
    )

    tx = tx[["date", "store_nbr", "transactions_lag1", "transactions_mean7"]]
    df = df.merge(tx, on=["date", "store_nbr"], how="left")
    df["transactions_lag1"] = df["transactions_lag1"].fillna(0)
    df["transactions_mean7"] = df["transactions_mean7"].fillna(0)
    return df


# ---------------------------------------------------------------------------
# Main feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, transactions: pd.DataFrame, holidays: pd.DataFrame,
                   oil: pd.DataFrame) -> pd.DataFrame:
    """
    Build all features on a full (or windowed) grid DataFrame.

    Parameters
    ----------
    df           : full grid with columns date, store_nbr, item_nbr, sales, onpromotion,
                   dcoilwtico, family, class, perishable, city, state, type, cluster
    transactions : raw transactions DataFrame
    holidays     : raw holidays_events DataFrame
    oil          : forward-filled oil DataFrame (date, dcoilwtico)

    Returns
    -------
    df with all feature columns added (no leakage).
    """
    df = df.copy()
    df = df.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

    # ---- Oil features ----
    oil_ma = oil.sort_values("date").copy()
    oil_ma["oil_ma7"] = oil_ma["dcoilwtico"].rolling(7, min_periods=1).mean()
    df = df.merge(oil_ma[["date", "oil_ma7"]], on="date", how="left")

    # ---- Calendar features ----
    df["day_of_week"] = df["date"].dt.dayofweek  # 0=Mon
    df["day_of_month"] = df["date"].dt.day
    df["is_payday"] = ((df["day_of_month"] == 15) | df["date"].dt.is_month_end).astype(int)
    df["is_month_start"] = (df["day_of_month"] <= 3).astype(int)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    # ---- Group key Series (defined after all merges that precede rolling ops) ----
    si_groups = [df["store_nbr"], df["item_nbr"]]

    # ---- Lag features (direct shift — no lambda) ----
    df["lag_7"] = df.groupby(si_groups)["sales"].shift(7)
    df["lag_14"] = df.groupby(si_groups)["sales"].shift(14)

    # ---- Store×item rolling features (shift by 1 then roll) ----
    sales_s1 = df.groupby(si_groups)["sales"].shift(1)

    df["sales_mean_7d"]   = _grp_roll(sales_s1, si_groups, 7,  "mean")
    df["sales_mean_14d"]  = _grp_roll(sales_s1, si_groups, 14, "mean")
    df["sales_mean_30d"]  = _grp_roll(sales_s1, si_groups, 30, "mean")
    df["sales_std_7d"]    = _grp_roll(sales_s1, si_groups, 7,  "std")
    df["sales_std_14d"]   = _grp_roll(sales_s1, si_groups, 14, "std")
    df["sales_median_7d"] = _grp_roll(sales_s1, si_groups, 7,  "median")

    # ---- Derived ----
    df["diff_7d_14d"]    = df["sales_mean_7d"] - df["sales_mean_14d"]
    df["ratio_7d_30d"]   = df["sales_mean_7d"] / (df["sales_mean_30d"] + 1e-3)
    df["sales_momentum"] = df["sales_mean_7d"] - df["sales_mean_14d"]

    # ---- Zero ratio ----
    zero_s1 = (sales_s1 == 0).astype(float)
    df["zero_ratio_7d"] = _grp_roll(zero_s1, si_groups, 7, "mean")

    # ---- Days since last nonzero (vectorized, called via transform) ----
    df["days_since_last_nonzero"] = (
        df.groupby(si_groups)["sales"].transform(_days_since_last_nonzero)
    )

    # ---- Promo features (store×item) ----
    df["onpromotion"] = df["onpromotion"].fillna(0)
    promo_s1 = df.groupby(si_groups)["onpromotion"].shift(1)

    df["promo_mean_7d"]  = _grp_roll(promo_s1, si_groups, 7,  "mean")
    df["promo_mean_14d"] = _grp_roll(promo_s1, si_groups, 14, "mean")
    df["promo_mean_30d"] = _grp_roll(promo_s1, si_groups, 30, "mean")
    df["promo_sum_7d"]   = _grp_roll(promo_s1, si_groups, 7,  "sum")
    df["promo_sum_14d"]  = _grp_roll(promo_s1, si_groups, 14, "sum")
    df["promo_sum_30d"]  = _grp_roll(promo_s1, si_groups, 30, "sum")
    df["promo_streak"]   = df.groupby(si_groups)["onpromotion"].transform(_promo_streak)

    # ---- Item-level rolling (global trend) ----
    item_groups = [df["item_nbr"]]
    item_s1 = df.groupby(item_groups)["sales"].shift(1)
    df["item_sales_mean_7d"]  = _grp_roll(item_s1, item_groups, 7,  "mean", n_group_levels=1)
    df["item_sales_mean_14d"] = _grp_roll(item_s1, item_groups, 14, "mean", n_group_levels=1)
    df["item_sales_mean_30d"] = _grp_roll(item_s1, item_groups, 30, "mean", n_group_levels=1)

    # ---- Store×family rolling ----
    if "family" in df.columns:
        sf_groups = [df["store_nbr"], df["family"]]
        sf_s1 = df.groupby(sf_groups)["sales"].shift(1)
        df["sf_sales_mean_7d"]  = _grp_roll(sf_s1, sf_groups, 7,  "mean")
        df["sf_sales_mean_14d"] = _grp_roll(sf_s1, sf_groups, 14, "mean")

        fam_groups = [df["family"]]
        fam_s1 = df.groupby(fam_groups)["sales"].shift(1)
        df["family_sales_mean_7d"] = _grp_roll(fam_s1, fam_groups, 7, "mean", n_group_levels=1)

    # ---- Store rolling (aggregate) ----
    store_groups = [df["store_nbr"]]
    store_s1 = df.groupby(store_groups)["sales"].shift(1)
    df["store_sales_mean_7d"] = _grp_roll(store_s1, store_groups, 7, "mean", n_group_levels=1)

    # ---- Transaction features ----
    df = build_transaction_features(df, transactions)

    # ---- Holiday features ----
    df = build_holiday_features(df, holidays)

    return df


# ---------------------------------------------------------------------------
# Feature list
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "lag_7", "lag_14",
    "sales_mean_7d", "sales_mean_14d", "sales_mean_30d",
    "sales_std_7d", "sales_std_14d",
    "sales_median_7d",
    "item_sales_mean_7d", "item_sales_mean_14d", "item_sales_mean_30d",
    "sf_sales_mean_7d", "sf_sales_mean_14d",
    "promo_mean_7d", "promo_mean_14d", "promo_mean_30d",
    "promo_sum_7d", "promo_sum_14d", "promo_sum_30d",
    "promo_streak",
    "days_since_last_nonzero", "zero_ratio_7d",
    "diff_7d_14d", "ratio_7d_30d", "sales_momentum",
    "dcoilwtico", "oil_ma7",
    "perishable",
    "day_of_week", "day_of_month",
    "is_payday", "is_month_start", "is_weekend",
    "transactions_lag1", "transactions_mean7",
    "store_sales_mean_7d", "family_sales_mean_7d",
    "is_holiday_local",
]

CAT_FEATURES = [
    "store_nbr", "item_nbr", "family", "type", "cluster", "class",
    "city", "state", "holiday_type",
]
