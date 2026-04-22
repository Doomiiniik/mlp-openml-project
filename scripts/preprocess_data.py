from pathlib import Path
import pandas as pd

from src.data import (
    download_openml_dataset,
    clean_data,
    split_data_stratified,
    fit_scaler,
    apply_scaler
)

def main():
    BASE_DIR = Path(__file__).resolve().parent.parent

    RAW_DIR = BASE_DIR / "data/raw"
    RAW_DIR.mkdir(exist_ok=True)

    RAW_PATH = RAW_DIR / "openml_6.csv"

    PROCESSED_DIR = BASE_DIR / "data/processed"
    PROCESSED_DIR.mkdir(exist_ok=True)

    # -----------------------------------------
    # 1. DOWNLOAD IF NOT EXISTS
    # -----------------------------------------
    if not RAW_PATH.exists():
        print("Raw data not found. Downloading from OpenML...")
        df, target_col = download_openml_dataset(
            dataset_id=6,          # Letter Recognition
            output_dir=RAW_DIR
        )
     
    else:
        print("Raw data found. Loading existing file...")
        df = pd.read_csv(RAW_PATH)
        target_col = "class"

    feature_cols = [c for c in df.columns if c != target_col]

    # -----------------------------------------
    # 2. CLEAN
    # -----------------------------------------
    print("Cleaning...")
    df_clean = clean_data(df)
    
    from sklearn.preprocessing import LabelEncoder
    import joblib

    # -----------------------------------------
    # 2.5 ENCODE TARGET COLUMN
    # -----------------------------------------
    print("Encoding target column...")

    le = LabelEncoder()
    df_clean[target_col] = le.fit_transform(df_clean[target_col])

    # Save encoder for inference
    joblib.dump(le, PROCESSED_DIR / "label_encoder.pkl")





    # -----------------------------------------
    # 3. SPLIT
    # -----------------------------------------
    print("Splitting...")
    train_df, val_df, test_df = split_data_stratified(df_clean, target_col)

    # -----------------------------------------
    # 4. SCALE
    # -----------------------------------------
    print("Scaling...")
    scaler = fit_scaler(train_df, feature_cols)
    train_scaled = apply_scaler(train_df, feature_cols, scaler)
    val_scaled = apply_scaler(val_df, feature_cols, scaler)
    test_scaled = apply_scaler(test_df, feature_cols, scaler)

    # -----------------------------------------
    # 5. SAVE
    # -----------------------------------------
    print("Saving processed data...")
    train_scaled.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_scaled.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_scaled.to_csv(PROCESSED_DIR / "test.csv", index=False)

    print("Done.")

if __name__ == "__main__":
    main()
