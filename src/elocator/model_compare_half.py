#!/usr/bin/env python3

"""
Compare two checkpoints (Model A and Model B) using the same dataset and architecture.
Model A => loaded from model_fixval_half.pth
Model B => loaded from model_fixval.pth
Generates MSE, MAE, and a double lift chart.
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import plotly.express as px
import pandas as pd
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Tuple, List

# ------------------------ Dataset (Model B Style) ------------------------

class ChessMoveDataset(Dataset):
    """
    Reads from a JSON file with records of the form:
        {
            "FEN": [float, float, ...]  # length=780
            "Accuracy": float
        }
    Accuracy is scaled by 100 in the dataset (0-100). We'll rescale to [0,1].
    """
    def __init__(self, json_file: str):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        features = torch.tensor(item['FEN'], dtype=torch.float32)  # shape [780]
        label = torch.tensor(item['Accuracy'] / 100.0, dtype=torch.float32)  
        return features, label

# ------------------------ Fully Connected Model (Model B Style) ------------------------

class ChessModel(nn.Module):
    """
    Fully connected architecture for an input feature length of 780.
    Uses BatchNorm and Dropout, with LeakyReLU activations, final Sigmoid output in [0,1].
    """
    def __init__(self, fen_size=780):
        super().__init__()
        self.fc1 = nn.Linear(fen_size, 4096)
        self.fc2 = nn.Linear(4096, 2056)
        self.fc3 = nn.Linear(2056, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)
        self.fc7 = nn.Linear(8, 1)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2056)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

        # Kaiming (He) initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc7.weight)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        # Output in [0,1]
        x = torch.sigmoid(self.fc7(x))
        return x

# ------------------------ Data Utilities ------------------------

def train_val_split(dataset: Dataset, split_ratio=0.8, seed=1337) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into train/val subsets with a given ratio and random seed.
    """
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def create_dataloaders(dataset: Dataset, batch_size=64, split_ratio=0.8, seed=1337) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/validation loaders from a dataset with a fixed batch_size and random seed.
    """
    train_ds, val_ds = train_val_split(dataset, split_ratio, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

# ------------------------ Training & Evaluation ------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr=0.001,
    weight_decay=1e-5,
    epochs=20,
    early_stop_rounds=5,
    model_save_path="model_temp.pth"
) -> None:
    """
    Generic training loop for a regression model with MSE loss, early stopping, 
    and learning-rate-scheduler on validation loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    stop_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(features).squeeze(-1)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            stop_count = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            stop_count += 1

        if stop_count >= early_stop_rounds:
            print(f"Early stopping triggered at epoch {epoch+1}.")
            break

def evaluate_model(model: nn.Module, data_loader: DataLoader) -> float:
    """Computes MSE on a given DataLoader."""
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            preds = model(features).squeeze(-1)
            loss = criterion(preds, labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def predict_model(model: nn.Module, data_loader: DataLoader) -> List[float]:
    """Returns predictions for all samples in data_loader, in order."""
    device = next(model.parameters()).device
    preds_list = []
    model.eval()
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            preds = model(features).squeeze(-1)
            preds_list.extend(preds.cpu().tolist())
    return preds_list

def get_actuals(data_loader: DataLoader) -> List[float]:
    """Returns the ground-truth labels in the data_loader, in order."""
    labels_list = []
    for _, labels in data_loader:
        labels_list.extend(labels.tolist())
    return labels_list

def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> dict:
    """
    Computes MSE and MAE for regression, returns them in a dict.
    """
    mse = sum((t - p)**2 for t, p in zip(y_true, y_pred)) / len(y_true)
    mae = sum(abs(t - p) for t, p in zip(y_true, y_pred)) / len(y_true)
    return {"MSE": mse, "MAE": mae}

# ------------------------ Double Lift Chart ------------------------

def create_double_lift_chart(predsA: List[float], predsB: List[float], actuals: List[float], n_deciles=10):
    """
    Creates a double lift chart:
      1) Create a DataFrame with predsA, predsB, actual
      2) Sort by average of (predsA + predsB)
      3) Decile the rows
      4) Compute average predsA, predsB, and actual in each decile
      5) Rebase actual => 1.0 per decile, so the lines for predsA, predsB,
         show their ratio to actual.
    """
    df = pd.DataFrame({"predA": predsA, "predB": predsB, "actual": actuals})
    df["avg_pred"] = (df["predA"] + df["predB"]) / 2
    df.sort_values(by="avg_pred", inplace=True, ignore_index=True)

    df["decile"] = pd.qcut(df["avg_pred"], n_deciles, labels=False, duplicates="drop")
    grouped = df.groupby("decile").mean(numeric_only=True)

    grouped["rebase_actual"] = 1.0
    grouped["rebase_A"] = grouped["predA"] / grouped["actual"]
    grouped["rebase_B"] = grouped["predB"] / grouped["actual"]

    fig = px.line(
        grouped[["rebase_A", "rebase_B", "rebase_actual"]],
        labels={"value": "Prediction (rebased)", "index": "Decile"}
    )
    fig.update_layout(
        title="Double Lift Chart (Prediction vs. Actual Rebased=1.0)",
        xaxis_title="Decile",
        yaxis_title="Ratio to Actual"
    )
    fig.show()

# ------------------------ Main Script ------------------------

def main():
    """
    1) Create one dataset from train.json (same features).
    2) Create two models (same architecture).
    3) Load different .pth files:
        - Model A => model_fixval_half.pth
        - Model B => model_fixval.pth
    4) Compare performance on the same validation split, create double lift chart.
    """

    torch.manual_seed(0)

    # ------------------ Dataset & DataLoaders ------------------
    dataset_path = "data/train.json"  # Adjust if needed
    dataset = ChessMoveDataset(dataset_path)

    # We'll use the *same* train/val loaders for reference if you want to measure
    # training/validation performance. If you only want predictions, you can skip training code.
    train_loader, val_loader = create_dataloaders(dataset, batch_size=64, split_ratio=0.8, seed=1337)

    # ------------------ Model A ------------------
    modelA = ChessModel(fen_size=780)
    # If you wanted to train from scratch, uncomment:
    # train_model(modelA, train_loader, val_loader, epochs=15, model_save_path="modelA_temp.pth")
    # Otherwise, load pre-trained weights
    modelA.load_state_dict(torch.load("data/model_fixval_half.pth", map_location="cpu"))

    # Predictions for Model A on validation set
    predsA = predict_model(modelA, val_loader)
    actualsA = get_actuals(val_loader)

    # ------------------ Model B ------------------
    modelB = ChessModel(fen_size=780)
    # If you wanted to train from scratch, uncomment:
    # train_model(modelB, train_loader, val_loader, epochs=15, model_save_path="modelB_temp.pth")
    # Otherwise, load pre-trained weights
    modelB.load_state_dict(torch.load("data/model_fixval.pth", map_location="cpu"))

    # Predictions for Model B on validation set
    predsB = predict_model(modelB, val_loader)
    actualsB = get_actuals(val_loader)

    # ------------------ Metrics ------------------
    # The actual labels should be identical in ordering and length, so let's confirm:
    if len(actualsA) != len(actualsB):
        raise ValueError("Mismatch in lengths between actualsA and actualsB!")
    # We'll just use one set of actual labels
    actuals = actualsA  

    metricsA = compute_regression_metrics(actuals, predsA)
    metricsB = compute_regression_metrics(actuals, predsB)

    print("=== Model A Results (model_fixval_half.pth) ===")
    print(f"   MSE: {metricsA['MSE']:.6f}")
    print(f"   MAE: {metricsA['MAE']:.6f}")

    print("=== Model B Results (model_fixval.pth) ===")
    print(f"   MSE: {metricsB['MSE']:.6f}")
    print(f"   MAE: {metricsB['MAE']:.6f}")

    # ------------------ Double Lift Chart ------------------
    create_double_lift_chart(predsA, predsB, actuals, n_deciles=10)
    print("Done. Close the chart window to finish.")

if __name__ == "__main__":
    main()
