# scripts/run_inference.py

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.models import build_mlp
from src.data import create_dataloader


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    TEST_PATH = BASE_DIR / "data/processed/test.csv"
    MODEL_PATH = BASE_DIR / "models/best_model.pt"
    CONFIG_PATH = BASE_DIR / "configs/best_config.json"

    if not TEST_PATH.exists():
        raise FileNotFoundError("Missing test.csv. Run preprocessing first.")

    print("Loading test data...")
    test_df = pd.read_csv(TEST_PATH)

    target_col = "class"
    feature_cols = [c for c in test_df.columns if c != target_col]

    # -----------------------------
    # Load best hyperparameters
    # -----------------------------
    import json
    with open(CONFIG_PATH) as f:
        best = json.load(f)

    # -----------------------------
    # Build model
    # -----------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_mlp(
        config={
            "hidden_layers": best["hidden_layers"],
            "dropout": best["dropout"]
        },
        input_dim=len(feature_cols),
        output_dim=test_df[target_col].nunique()
    ).to(device)

    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # -----------------------------
    # Create dataloader
    # -----------------------------
    test_loader = create_dataloader(
        test_df, feature_cols, target_col,
        batch_size=best["batch_size"], shuffle=False
    )

    # -----------------------------
    # Inference
    # -----------------------------
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            logits = model(X)
            preds = logits.argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(y.numpy())


    # -----------------------------
    # Save predictions
    # -----------------------------
    out_df = pd.DataFrame({
        "true": all_targets,
        "pred": all_preds
    })

    # -----------------------------
    # Save predictions to outputs/
    # -----------------------------
    outputs_dir = BASE_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    out_path = outputs_dir / "predictions.csv"
    out_df.to_csv(out_path, index=False)

    print(f"\nPredictions saved to {out_path}")


if __name__ == "__main__":
    main()
