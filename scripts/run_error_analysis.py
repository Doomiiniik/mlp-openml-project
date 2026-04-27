# scripts/run_error_analysis.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    pred_path = BASE_DIR / "outputs" / "predictions.csv"

    if not pred_path.exists():
        raise FileNotFoundError("Missing predictions.csv. Run inference first.")

    df = pd.read_csv(pred_path)


    # Load label encoder
    le_path = BASE_DIR / "data" / "processed" / "label_encoder.pkl"
    import joblib
    le = joblib.load(le_path)

    # Decode numeric labels to letters
    df["true_letter"] = le.inverse_transform(df["true"])
    df["pred_letter"] = le.inverse_transform(df["pred"])

    y_true = df["true_letter"].values
    y_pred = df["pred_letter"].values
















    # -----------------------------
    # Per-class metrics
    # -----------------------------
    report = classification_report(y_true, y_pred, output_dict=True)

    per_class_f1 = {
    cls: round(report[cls]["f1-score"], 4)
    for cls in report.keys()
    if cls not in ["accuracy", "macro avg", "weighted avg"]
}


    # Sort classes by difficulty (lowest F1 first)
    hardest = sorted(per_class_f1.items(), key=lambda x: x[1])[:5]

    df_errors = df[df["true_letter"] != df["pred_letter"]]
    sample_errors = df_errors[["true_letter", "pred_letter"]].sample(min(20, len(df_errors)), random_state=42)

    # -----------------------------
    # Save analysis
    # -----------------------------
    out = {
        "num_samples": len(df),
        "num_errors": len(df_errors),
        "error_rate": round(len(df_errors) / len(df), 4),
        "per_class_f1": per_class_f1,
        "hardest_classes": hardest,
        "example_errors": sample_errors.to_dict(orient="records"),
    }

    out_path = BASE_DIR / "outputs" / "error_analysis.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=4)

    # -----------------------------
    # Print summary
    # -----------------------------
    print("\n=== ERROR ANALYSIS SUMMARY ===")
    print(f"Total samples: {out['num_samples']}")
    print(f"Errors: {out['num_errors']} ({out['error_rate']*100:.2f}%)")

    print("\nHardest classes (lowest F1):")
    for cls, f1 in hardest:
        print(f"  Class {cls}: F1 = {f1}")

    print("\nExample misclassifications:")
    print(sample_errors.head())

    print(f"\nFull report saved to: {out_path}")


if __name__ == "__main__":
    main()
