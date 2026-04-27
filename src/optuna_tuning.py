import optuna
import torch
from sklearn.model_selection import StratifiedKFold
from src.models import build_mlp
from src.train import train_model, validate
from src.data import create_dataloader

def objective(trial, df, feature_cols, target_col, device):

    # --- przestrzeń hiperparametrów ---
    hidden_layers = trial.suggest_categorical(
        "hidden_layers",
        [
            [64],
            [128],
            [256],
            [64, 32],
            [128, 64],
            [256, 128],
            [128, 128, 64]
        ]
    )

    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    grad_clip = trial.suggest_float("grad_clip", 0.5, 5.0)

    # --- 5-fold CV ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    val_losses = []

    for train_idx, val_idx in skf.split(df, df[target_col]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        train_loader = create_dataloader(train_df, feature_cols, target_col, batch_size, shuffle=True)
        val_loader = create_dataloader(val_df, feature_cols, target_col, batch_size, shuffle=False)

        model = build_mlp(
            config={"hidden_layers": hidden_layers, "dropout": dropout},
            input_dim=len(feature_cols),
            output_dim=df[target_col].nunique()
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            epochs=10,
            grad_clip=grad_clip,
            log_dir=None,          
            save_path="temp.pt"    
        )

        val_losses.append(best_val_loss)

    return sum(val_losses) / len(val_losses)



def run_optuna(df, feature_cols, target_col, device, n_trials=20):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, df, feature_cols, target_col, device),
        n_trials=n_trials
    )
    return study


import json

def save_best_config(study, path="configs/best_config.json"):
    with open(path, "w") as f:
        json.dump(study.best_params, f, indent=4)

