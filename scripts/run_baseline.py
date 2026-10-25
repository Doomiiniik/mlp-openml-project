# scripts/run_baseline.py
import time
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    BASE_DIR = Path(__file__).resolve().parent.parent
    target_col = "class"

    train_df = pd.read_csv(BASE_DIR / "data/processed/train.csv")
    test_df = pd.read_csv(BASE_DIR / "data/processed/test.csv")

    X_train, y_train = train_df.drop(columns=target_col), train_df[target_col]
    X_test, y_test = test_df.drop(columns=target_col), test_df[target_col]

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
    }

    rows = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        rows.append({
            "model": name,
            "test_accuracy": round(acc, 4),
            "fit_seconds": round(time.time() - start, 1),
        })

    results = pd.DataFrame(rows)
    print(results.to_string(index=False))
    results.to_csv(BASE_DIR / "outputs" / "baseline_results.csv", index=False)


if __name__ == "__main__":
    main()