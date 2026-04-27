"""Data loading and merging for Favorita grocery sales forecasting."""

import os
import pandas as pd
from config import DATA_DIR

_TRAIN_CUTOFF = "2017-06-11"   # max_date(2017-08-15) - 65 days = window_start
_CHUNK_SIZE = 2_000_000


def load_raw_data(data_dir: str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Load all CSV files. Train is streamed in chunks and date-filtered."""
    non_train = {
        "test": "test.csv",
        "stores": "stores.csv",
        "items": "items.csv",
        "oil": "oil.csv",
        "holidays": "holidays_events.csv",
        "transactions": "transactions.csv",
    }
    date_files = {"test", "oil", "holidays", "transactions"}
    dfs = {}
    for key, fname in non_train.items():
        path = os.path.join(data_dir, fname)
        parse = ["date"] if key in date_files else []
        dfs[key] = pd.read_csv(path, parse_dates=parse, low_memory=False)

    # Train: 125M rows — read in chunks, keep only the window we need
    train_path = os.path.join(data_dir, "train.csv")
    cutoff = pd.Timestamp(_TRAIN_CUTOFF)
    chunks = []
    print(f"  Reading train.csv in chunks (cutoff={_TRAIN_CUTOFF}) ...")
    for chunk in pd.read_csv(train_path, parse_dates=["date"],
                              chunksize=_CHUNK_SIZE, low_memory=False):
        chunk = chunk[chunk["date"] >= cutoff]
        if len(chunk):
            chunks.append(chunk)
    dfs["train"] = pd.concat(chunks, ignore_index=True)
    print(f"  Train rows after filter: {len(dfs['train']):,}")

    return dfs


def prepare_oil(oil: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill oil prices over the entire date range before any split."""
    oil = oil.copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.sort_values("date").set_index("date")
    full_idx = pd.date_range(oil.index.min(), oil.index.max(), freq="D")
    oil = oil.reindex(full_idx).ffill().bfill().reset_index()
    oil.columns = ["date", "dcoilwtico"]
    return oil


def prepare_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Return transactions with date parsed."""
    tx = transactions.copy()
    tx["date"] = pd.to_datetime(tx["date"])
    return tx


def merge_store_item(df: pd.DataFrame, stores: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
    """Merge store and item metadata into train/test."""
    stores = stores[["store_nbr", "city", "state", "type", "cluster"]].copy()
    items = items[["item_nbr", "family", "class", "perishable"]].copy()
    df = df.merge(stores, on="store_nbr", how="left")
    df = df.merge(items, on="item_nbr", how="left")
    return df


def load_and_merge(data_dir: str = DATA_DIR):
    """
    Main entry point.

    Returns
    -------
    train : pd.DataFrame  — train with store/item meta + oil + transactions
    test  : pd.DataFrame  — test  with store/item meta + oil + transactions
    oil   : pd.DataFrame  — forward-filled oil (full range)
    transactions : pd.DataFrame
    holidays : pd.DataFrame
    """
    raw = load_raw_data(data_dir)

    train = raw["train"].copy()
    test = raw["test"].copy()

    if "unit_sales" in train.columns:
        train = train.rename(columns={"unit_sales": "sales"})

    # Clip returns (negative unit_sales) to 0
    train["sales"] = train["sales"].clip(lower=0)

    stores = raw["stores"]
    items = raw["items"]

    train["date"] = pd.to_datetime(train["date"])
    test["date"] = pd.to_datetime(test["date"])

    # Forward-fill oil over entire range *before* any split
    oil = prepare_oil(raw["oil"])

    # Merge store and item metadata
    train = merge_store_item(train, stores, items)
    test = merge_store_item(test, stores, items)

    # Merge oil by date (no leakage — oil is exogenous)
    train = train.merge(oil, on="date", how="left")
    test = test.merge(oil, on="date", how="left")

    transactions = prepare_transactions(raw["transactions"])
    holidays = raw["holidays"].copy()
    holidays["date"] = pd.to_datetime(holidays["date"])

    return train, test, oil, transactions, holidays
