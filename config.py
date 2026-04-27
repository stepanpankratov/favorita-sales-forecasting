DATA_DIR = "./favorita-grocery-sales-forecasting"
RESULTS_DIR = "./results"
SEED = 42
N_DAYS_HISTORY = 66
N_VAL_DAYS = 16
N_FORECAST = 16
BASELINE_SAMPLE = 1000
FULL_GRID = True  # build full date×store×item grid
GRID_MODE = "sparse"  # "full" = cross-join all dates×pairs; "sparse" = existing records only

CAT_FEATURES = [
    "store_nbr",
    "item_nbr",
    "family",
    "type",
    "cluster",
    "class",
    "city",
    "state",
]

CATBOOST_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "loss_function": "RMSE",
    "eval_metric": "RMSE",
    "early_stopping_rounds": 50,
    "random_seed": SEED,
    "verbose": 100,
}

NN_PARAMS = {
    "lr": 1e-3,
    "batch_size": 4096,
    "epochs": 20,
    "dropout": 0.2,
}
NN_SAMPLE_PAIRS = 5000  # subsample store-item pairs for NN (None = use all)
