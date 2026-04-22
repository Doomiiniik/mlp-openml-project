# src/data.py
from pathlib import Path
import openml
import pandas as pd

def download_openml_dataset(dataset_id: int, output_dir: str = "data/raw"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = openml.datasets.get_dataset(dataset_id)
    X, y, _, _ = dataset.get_data(dataset_format="dataframe",
                                  target=dataset.default_target_attribute)

    df = X.copy()
    df[dataset.default_target_attribute] = y

    path = output / f"openml_{dataset_id}.csv"
    df.to_csv(path, index=False)

    return df, dataset.default_target_attribute


def clean_data(df):
    """Remove duplicates and reset index."""
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df_clean



from sklearn.model_selection import train_test_split

def split_data_stratified(df, target, test_size=0.2, val_size=0.1, random_state=42):
    """Stratified train/val/test split."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[target], random_state=random_state
    )

    train_df, val_df = train_test_split(
        train_df, test_size=val_size, stratify=train_df[target], random_state=random_state
    )

    return train_df, val_df, test_df

from sklearn.preprocessing import StandardScaler
import pandas as pd

def fit_scaler(train_df, numeric_cols):
    """Fit StandardScaler on training data only."""
    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])
    return scaler


def apply_scaler(df, numeric_cols, scaler):
    """Apply fitted scaler to a dataframe."""
    df_scaled = df.copy()
    df_scaled[numeric_cols] = scaler.transform(df[numeric_cols])
    return df_scaled


import torch
from torch.utils.data import Dataset

class TabularDataset(Dataset):
    def __init__(self, df, feature_cols, target_col):
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(df[target_col].values, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

from torch.utils.data import DataLoader

def create_dataloader(df, feature_cols, target_col, batch_size, shuffle=True):
    dataset = TabularDataset(df, feature_cols, target_col)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)





