import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layers, output_dim, dropout=0.0):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_layers:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)




def build_mlp(config, input_dim, output_dim):
    hidden_layers = config["hidden_layers"]
    dropout = config.get("dropout", 0.0)

    return MLP(
        input_dim=input_dim,
        hidden_layers=hidden_layers,
        output_dim=output_dim,
        dropout=dropout
    )
