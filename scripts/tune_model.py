# scripts/tune_model.py

import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.optuna_tuning import run_optuna


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(42)

    BASE_DIR = Path(__file__).resolve().parent.parent

    # -----------------------------------------
    # 1. LOAD PROCESSED TRAIN DATA
    # -----------------------------------------
    TRAIN_PATH = BASE_DIR / "data/processed/train.csv"
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Processed train data not found at {TRAIN_PATH}. "
            "Run scripts/preprocess_data.py first."
        )

    print("Loading processed training data...")
    df = pd.read_csv(TRAIN_PATH)

    target_col = "class"
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in train.csv")

    feature_cols = [c for c in df.columns if c != target_col]

    # -----------------------------------------
    # 2. DEVICE
    # -----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------
    # 3. RUN OPTUNA
    # -----------------------------------------
    print("Starting Optuna hyperparameter search...")
    study = run_optuna(
        df=df,
        feature_cols=feature_cols,
        target_col=target_col,
        device=device,
        n_trials=20
    )

    print("Optuna finished.")
    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)

    # -----------------------------------------
    # 4. SAVE BEST CONFIG
    # -----------------------------------------
    CONFIG_DIR = BASE_DIR / "configs"
    CONFIG_DIR.mkdir(exist_ok=True)

    config_path = CONFIG_DIR / "best_config.json"
    with open(config_path, "w") as f:
        json.dump(study.best_params, f, indent=4)

    print(f"Best config saved to {config_path}")

    # Optional: save trials
    study.trials_dataframe().to_csv(CONFIG_DIR / "optuna_trials.csv", index=False)
    print("Trials saved to optuna_trials.csv")


if __name__ == "__main__":
    main()
