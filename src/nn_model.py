"""MLP + Embeddings model (MIMO: one pass → 16-step forecast)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import LabelEncoder

from config import N_FORECAST, NN_PARAMS, SEED
from src.metrics import evaluate
import src.features as F


# ---------------------------------------------------------------------------
# Encoders
# ---------------------------------------------------------------------------

class CatEncoder:
    """Label-encoder with UNK=0 bucket for unseen categories."""

    def __init__(self):
        self.encoders: dict[str, LabelEncoder] = {}
        self.n_cats: dict[str, int] = {}

    def fit(self, df: pd.DataFrame, cat_cols: list[str]):
        for col in cat_cols:
            le = LabelEncoder()
            vals = df[col].fillna("UNK").astype(str)
            le.fit(["UNK"] + sorted(vals.unique().tolist()))
            self.encoders[col] = le
            self.n_cats[col] = len(le.classes_) + 1  # +1 for extra UNK

    def transform(self, df: pd.DataFrame, cat_cols: list[str]) -> np.ndarray:
        result = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
        for i, col in enumerate(cat_cols):
            le = self.encoders[col]
            vals = df[col].fillna("UNK").astype(str)
            known = set(le.classes_)
            encoded = vals.map(lambda v, _le=le, _k=known: _le.transform([v])[0] if v in _k else 0)
            result[:, i] = encoded.values
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MIMODataset(Dataset):
    """
    Each sample: anchor row  →  X_num, X_cat, y (N_FORECAST steps ahead)

    Vectorized: uses groupby().shift(-h) to look ahead h steps.

    Parameters
    ----------
    grid         : full grid (train+val) sorted by (store, item, date)
    anchor_mask  : boolean mask — which rows may serve as anchors
                   (targets are looked up from the entire grid regardless)
    """

    def __init__(self, grid: pd.DataFrame, cat_encoder: CatEncoder,
                 cat_cols: list[str], num_cols: list[str],
                 horizon: int = N_FORECAST, anchor_mask: pd.Series | None = None):
        grid = grid.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

        si_groups = ["store_nbr", "item_nbr"]

        # Compute future targets via grouped negative-shift (on entire grid)
        target_cols = {}
        valid_mask = pd.Series(True, index=grid.index)
        for h in range(1, horizon + 1):
            fwd = grid.groupby(si_groups)["target"].shift(-h)
            target_cols[h] = fwd
            valid_mask &= fwd.notna()

        # Restrict anchors
        if anchor_mask is not None:
            # anchor_mask was built before sort/reset, so re-align by position
            if len(anchor_mask) == len(grid):
                valid_mask &= anchor_mask.values
            else:
                valid_mask &= anchor_mask.reindex(grid.index, fill_value=False).values

        valid_idx = grid.index[valid_mask]
        valid_grid = grid.loc[valid_idx].copy()

        if len(valid_grid) == 0:
            raise ValueError("No valid MIMO samples found in grid.")

        # Stack targets: shape (n_samples, horizon)
        self.targets = np.stack(
            [target_cols[h].loc[valid_idx].values for h in range(1, horizon + 1)],
            axis=1,
        ).astype(np.float32)

        # Numeric features
        for col in num_cols:
            if col not in valid_grid.columns:
                valid_grid[col] = 0.0
            valid_grid[col] = valid_grid[col].fillna(0).astype(float)

        self.X_num = valid_grid[num_cols].values.astype(np.float32)
        self.X_cat = cat_encoder.transform(valid_grid, cat_cols).astype(np.int64)
        # Keep anchor metadata for unrolling predictions later
        self.meta = valid_grid[["store_nbr", "item_nbr", "date"]].reset_index(drop=True)

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.X_num[idx]),
            torch.tensor(self.X_cat[idx]),
            torch.tensor(self.targets[idx]),
        )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MIMONet(nn.Module):
    """MLP with embeddings, MIMO output (N_FORECAST steps)."""

    def __init__(self, n_num: int, cat_dims: list[tuple[int, int]],
                 hidden: list[int] = (512, 256, 128),
                 dropout: float = 0.2, output_dim: int = N_FORECAST):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, emb_dim) for n_cats, emb_dim in cat_dims
        ])
        emb_total = sum(d for _, d in cat_dims)
        in_dim = n_num + emb_total

        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat([x_num] + embs, dim=1)
        return self.mlp(x)


def _embedding_dim(n_cats: int) -> int:
    return min(50, (n_cats + 1) // 2)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_nn(full_grid: pd.DataFrame, val_start: pd.Timestamp) -> tuple:
    """
    Train MIMO MLP.

    Parameters
    ----------
    full_grid : feature-augmented grid covering both train and val periods
    val_start : first date of validation period

    Returns
    -------
    model, cat_encoder, cat_cols, num_cols, val_metrics, val_pred_df
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    cat_cols = [c for c in F.CAT_FEATURES if c in full_grid.columns]
    num_cols = [c for c in F.NUMERIC_FEATURES if c in full_grid.columns] + ["onpromotion"]
    num_cols = list(dict.fromkeys(num_cols))  # deduplicate

    cat_encoder = CatEncoder()
    cat_encoder.fit(full_grid, cat_cols)

    # Sort once for consistent shift semantics
    full_grid = full_grid.sort_values(["store_nbr", "item_nbr", "date"]).reset_index(drop=True)

    # Train anchors: date < val_start (future targets may extend into val — that's fine,
    # we're predicting ahead from a past snapshot)
    # Val anchors: last N_FORECAST training days whose futures land in the val period
    val_anchor_cutoff = val_start - pd.Timedelta(days=N_FORECAST)
    train_mask = full_grid["date"] < val_anchor_cutoff
    val_mask = (full_grid["date"] >= val_anchor_cutoff) & (full_grid["date"] < val_start)

    n_train_anchors = train_mask.sum()
    n_val_anchors = val_mask.sum()
    print(f"  MIMO anchors: train={n_train_anchors:,}  val={n_val_anchors:,}")

    train_ds = MIMODataset(full_grid, cat_encoder, cat_cols, num_cols, anchor_mask=train_mask)
    val_ds = MIMODataset(full_grid, cat_encoder, cat_cols, num_cols, anchor_mask=val_mask)

    print(f"  MIMO samples: train={len(train_ds):,}  val={len(val_ds):,}")

    train_loader = DataLoader(train_ds, batch_size=NN_PARAMS["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=NN_PARAMS["batch_size"] * 2, shuffle=False)

    cat_dims = [(cat_encoder.n_cats[c], _embedding_dim(cat_encoder.n_cats[c])) for c in cat_cols]
    model = MIMONet(n_num=len(num_cols), cat_dims=cat_dims, dropout=NN_PARAMS["dropout"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=NN_PARAMS["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None

    for epoch in range(NN_PARAMS["epochs"]):
        model.train()
        for x_num, x_cat, y in train_loader:
            x_num, x_cat, y = x_num.to(device), x_cat.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x_num, x_cat)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x_num, x_cat, y in val_loader:
                x_num, x_cat = x_num.to(device), x_cat.to(device)
                pred = model(x_num, x_cat)
                all_preds.append(pred.cpu().numpy())
                all_targets.append(y.numpy())

        val_pred_log = np.vstack(all_preds)
        val_tgt_log = np.vstack(all_targets)
        val_pred_sales = np.expm1(val_pred_log).clip(0)
        val_true_sales = np.expm1(val_tgt_log).clip(0)

        nwrmsle_val = _compute_flat_nwrmsle(val_true_sales, val_pred_sales)
        print(f"  Epoch {epoch+1}/{NN_PARAMS['epochs']}  val_nwrmsle={nwrmsle_val:.4f}")

        if nwrmsle_val < best_val:
            best_val = nwrmsle_val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final val predictions
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_num, x_cat, y in val_loader:
            x_num, x_cat = x_num.to(device), x_cat.to(device)
            all_preds.append(model(x_num, x_cat).cpu().numpy())

    final_pred = np.expm1(np.vstack(all_preds)).clip(0)
    final_true = np.expm1(val_ds.targets).clip(0)

    val_metrics = {
        "nwrmsle": _compute_flat_nwrmsle(final_true, final_pred),
    }

    # Unroll MIMO predictions into per-(store, item, date) format for ensemble
    val_rows = []
    for h in range(N_FORECAST):
        chunk = val_ds.meta.copy()
        chunk["date"] = chunk["date"] + pd.Timedelta(days=h + 1)  # target date
        chunk["pred"] = final_pred[:, h]
        val_rows.append(chunk)
    val_pred_df = pd.concat(val_rows, ignore_index=True)
    # Keep only val period and average overlapping predictions
    val_pred_df = val_pred_df[val_pred_df["date"] >= val_start]
    val_pred_df = (val_pred_df.groupby(["store_nbr", "item_nbr", "date"], as_index=False)
                   .agg(pred=("pred", "mean")))

    return model, cat_encoder, cat_cols, num_cols, val_metrics, val_pred_df


def _compute_flat_nwrmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Flatten 2D arrays and compute NWRMSLE (unit weights)."""
    yt = y_true.ravel()
    yp = y_pred.ravel()
    log_diff = np.log1p(yp) - np.log1p(yt)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def predict_nn(model: MIMONet, cat_encoder: CatEncoder,
               cat_cols: list[str], num_cols: list[str],
               grid: pd.DataFrame) -> pd.DataFrame:
    """
    Predict on the last-date snapshot (for test inference).

    Returns
    -------
    pd.DataFrame with store_nbr, item_nbr, horizon_day (1..N_FORECAST), pred
    """
    device = next(model.parameters()).device
    last_date = grid["date"].max()
    anchor = grid[grid["date"] == last_date].copy()

    for col in num_cols:
        if col not in anchor.columns:
            anchor[col] = 0.0
        anchor[col] = anchor[col].fillna(0).astype(float)

    X_num = torch.tensor(anchor[num_cols].values.astype(np.float32)).to(device)
    X_cat = torch.tensor(cat_encoder.transform(anchor, cat_cols).astype(np.int64)).to(device)

    model.eval()
    with torch.no_grad():
        pred_log = model(X_num, X_cat).cpu().numpy()

    pred = np.expm1(pred_log).clip(0)  # shape: (n_pairs, N_FORECAST)

    rows = []
    for h in range(N_FORECAST):
        chunk = anchor[["store_nbr", "item_nbr"]].copy()
        chunk["horizon_day"] = h + 1
        chunk["date"] = last_date + pd.Timedelta(days=h + 1)
        chunk["pred"] = pred[:, h]
        rows.append(chunk)

    return pd.concat(rows, ignore_index=True)
