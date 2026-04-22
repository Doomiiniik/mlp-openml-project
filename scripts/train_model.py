# scripts/train_model.py

import json
import random
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from src.models import build_mlp
from src.data import create_dataloader
from src.train import train_final_model


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
    # 1. LOAD PROCESSED TRAIN + VAL DATA
    # -----------------------------------------
    TRAIN_PATH = BASE_DIR / "data/processed/train.csv"
    VAL_PATH = BASE_DIR / "data/processed/val.csv"

    if not TRAIN_PATH.exists() or not VAL_PATH.exists():
        raise FileNotFoundError(
            "Processed data not found. Run scripts/preprocess_data.py first."
        )

    print("Loading processed data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)

    target_col = "class"
    feature_cols = [c for c in train_df.columns if c != target_col]

    # -----------------------------------------
    # 2. MERGE TRAIN + VAL FOR FINAL TRAINING
    # -----------------------------------------
    full_df = pd.concat([train_df, val_df], ignore_index=True)
    print(f"Final training dataset size: {len(full_df)} samples")

    # -----------------------------------------
    # 3. LOAD BEST CONFIG FROM OPTUNA
    # -----------------------------------------
    CONFIG_PATH = BASE_DIR / "configs/best_config.json"
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            "best_config.json not found. Run scripts/tune_model.py first."
        )

    with open(CONFIG_PATH) as f:
        best = json.load(f)

    print("Loaded best hyperparameters:", best)

    # -----------------------------------------
    # 4. DEVICE
    # -----------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------
    # 5. BUILD MODEL
    # -----------------------------------------
    model = build_mlp(
        config={
            "hidden_layers": best["hidden_layers"],
            "dropout": best["dropout"]
        },
        input_dim=len(feature_cols),
        output_dim=train_df[target_col].nunique()
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=best["lr"])

    # -----------------------------------------
    # 6. FINAL TRAINING DATALOADER (NO VALIDATION)
    # -----------------------------------------
    train_loader = create_dataloader(
        full_df, feature_cols, target_col,
        batch_size=best["batch_size"], shuffle=True
    )

    # -----------------------------------------
    # 7. FINAL TRAINING
    # -----------------------------------------
    print("Starting final training...")

    MODELS_DIR = BASE_DIR / "models"
    MODELS_DIR.mkdir(exist_ok=True)

    best_model_path = MODELS_DIR / "best_model.pt"
    log_dir = BASE_DIR / "runs/final"

    train_final_model(
        model=model,
        train_loader=train_loader,
        val_loader=None,          # no validation in final training
        optimizer=optimizer,
        device=device,
        epochs=40,
        grad_clip=best["grad_clip"],
        save_path=best_model_path,
        log_dir=str(log_dir)
    )

    print(f"Final model saved to {best_model_path}")


if __name__ == "__main__":
    main()
