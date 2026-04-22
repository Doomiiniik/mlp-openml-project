import torch

def prepare_features(df, feature_cols):
    X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
    return X


def predict_proba(model, X, device="cpu"):
    model.eval()
    with torch.no_grad():
        X = X.to(device)
        logits = model(X)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu()



import numpy as np

def predict(model, X, device="cpu", class_mapping=None):
    probs = predict_proba(model, X, device)
    preds = probs.argmax(dim=1).numpy()

    if class_mapping is not None:
        preds = [class_mapping[p] for p in preds]

    return preds


def load_model_for_inference(model_class, config, input_dim, output_dim, path, device="cpu"):
    model = model_class(
        input_dim=input_dim,
        hidden_layers=config["hidden_layers"],
        output_dim=output_dim,
        dropout=config.get("dropout", 0.0)
    )
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model



def run_inference(df, feature_cols, model, device="cpu", class_mapping=None):
    X = prepare_features(df, feature_cols)
    preds = predict(model, X, device, class_mapping)
    return preds
