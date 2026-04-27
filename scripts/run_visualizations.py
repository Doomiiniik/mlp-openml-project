# scripts/run_visualizations.py

import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import confusion_matrix

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    # Paths
    pred_path = BASE_DIR / "outputs" / "predictions.csv"
    error_path = BASE_DIR / "outputs" / "error_analysis.json"
    le_path = BASE_DIR / "data" / "processed" / "label_encoder.pkl"

    # Output directory for plots
    plots_dir = BASE_DIR / "outputs" / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(pred_path)
    with open(error_path) as f:
        error_data = json.load(f)
    le = joblib.load(le_path)

    # Decode labels
    df["true_letter"] = le.inverse_transform(df["true"])
    df["pred_letter"] = le.inverse_transform(df["pred"])

    # -----------------------------
    # 1. Confusion Matrix (PNG)
    # -----------------------------
    cm = confusion_matrix(df["true_letter"], df["pred_letter"], labels=le.classes_)

    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Confusion Matrix (A–Z)", fontsize=18)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(plots_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    # -----------------------------
    # 2. F1-score Bar Chart
    # -----------------------------
    f1_scores = error_data["per_class_f1"]
    letters = list(f1_scores.keys())
    values = list(f1_scores.values())

    sorted_pairs = sorted(zip(letters, values), key=lambda x: x[1])
    letters_sorted, values_sorted = zip(*sorted_pairs)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=list(letters_sorted), y=list(values_sorted), palette="viridis")
    plt.title("F1-score per Class (Sorted)", fontsize=18)
    plt.xlabel("Letter")
    plt.ylabel("F1-score")
    plt.ylim(0.85, 1.0)
    plt.tight_layout()
    plt.savefig(plots_dir / "f1_scores.png", dpi=300)
    plt.close()

    # -----------------------------
    # 3. Hardest Classes Bar Chart
    # -----------------------------
    hardest = error_data["hardest_classes"]
    hardest_letters = [x[0] for x in hardest]
    hardest_values = [x[1] for x in hardest]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=hardest_letters, y=hardest_values, palette="rocket")
    plt.title("Hardest Classes (Lowest F1)", fontsize=18)
    plt.xlabel("Letter")
    plt.ylabel("F1-score")
    plt.ylim(0.85, 1.0)
    plt.tight_layout()
    plt.savefig(plots_dir / "hardest_classes.png", dpi=300)
    plt.close()

    # -----------------------------
    # 4. Error Pairs Heatmap (True vs Pred)
    # -----------------------------
    error_matrix = pd.crosstab(df["true_letter"], df["pred_letter"])

    plt.figure(figsize=(14, 12))
    sns.heatmap(error_matrix, cmap="magma", xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title("Error Pair Heatmap (True vs Pred)", fontsize=18)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(plots_dir / "error_pairs.png", dpi=300)
    plt.close()

    print("\n=== VISUALIZATIONS GENERATED ===")
    print(f"Saved to: {plots_dir}")


if __name__ == "__main__":
    main()
