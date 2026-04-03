#!/usr/bin/env python3

"""
Script to train or load two different chess-accuracy models (Model A and Model B) on two distinct feature sets
but matching labels, then compare predictions on a validation set. Also generates a double lift chart.
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
import math

# ------------------------ Model A Code ------------------------

class ChessMoveDatasetA(Dataset):
    """
    PyTorch Dataset for Model A, reading from a .pt file with keys:
        {
            'FENs': torch.Tensor of shape (N, 18, 8, 8),
            'Metadata': list of dicts, where each dict has 'Accuracy' key
        }
    """
    def __init__(self, tensor_file_path: str):
        data = torch.load(tensor_file_path)
        self.features = data['FENs']
        # The labels are the same length as features; divide by 100 so they are in [0,1]
        self.labels = torch.tensor(
            [item['Accuracy'] for item in data['Metadata']], dtype=torch.float32
        ) / 100.0

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx: int):
        return self.features[idx], self.labels[idx]


def init_kaiming(m: nn.Module):
    """
    Utility function to apply Kaiming normal initialization
    to Conv2d or Linear layers, and set BatchNorm2d parameters.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


class SubNetworkA(nn.Module):
    """
    Sub-network A (part of an ensemble).
    """
    def __init__(self, in_channels=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout(0.3)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Kaiming init
        self.apply(init_kaiming)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = self.dropout2(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.01)
        x = self.gap(x).view(x.size(0), -1)
        return x


class SubNetworkB(nn.Module):
    """
    Sub-network B (part of an ensemble).
    """
    def __init__(self, in_channels=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(32 * 8 * 8, 64)

        # Kaiming init
        self.apply(init_kaiming)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.01)
        x = self.dropout1(x)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.01)
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class EnsembleChessModel(nn.Module):
    """
    Model A: An ensemble of SubNetworkA and SubNetworkB, merged with FC layers.
    """
    def __init__(self):
        super().__init__()
        self.subA = SubNetworkA()
        self.subB = SubNetworkB()
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        # Kaiming init
        self.apply(init_kaiming)

    def forward(self, x):
        embA = self.subA(x)
        embB = self.subB(x)
        x = torch.cat([embA, embB], dim=-1)
        return self.fc(x)

# ------------------------ Model B Code ------------------------

class ChessMoveDatasetB(Dataset):
    """
    PyTorch Dataset for Model B, reading from a JSON file with keys:
        [
            {
                "FEN": [float, float, ...],  # length 780,
                "Accuracy": <float>
            },
            ...
        ]
    """
    def __init__(self, json_file: str):
        with open(json_file, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        item = self.data[idx]
        # Convert the FEN (length 780) to a torch tensor
        features = torch.tensor(item['FEN'], dtype=torch.float32)
        # Divide Accuracy by 100 to get range [0,1]
        label = torch.tensor(item['Accuracy'] / 100.0, dtype=torch.float32)
        return features, label


class ChessModelB(nn.Module):
    """
    Model B: A deep fully connected model for input dimension = 780.
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

        # BatchNorm
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2056)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)

        # Dropout
        self.dropout = nn.Dropout(0.5)

        # Initialize weights
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc7.weight)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), 0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), 0.01)
        x = F.leaky_relu(self.fc6(x), 0.01)
        # Sigmoid to keep final output in range [0,1].
        x = torch.sigmoid(self.fc7(x))
        return x

# ------------------------ Utilities ------------------------

def train_val_split(
    dataset: Dataset, split_ratio: float = 0.8, seed: int = 1337
) -> Tuple[Dataset, Dataset]:
    """
    Splits a Dataset into train/val subsets with given split ratio and random seed.
    """
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)


def create_dataloaders(
    dataset: Dataset, batch_size: int = 64, split_ratio: float = 0.8, seed: int = 1337
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits dataset into train/val, and returns corresponding DataLoaders.
    """
    train_ds, val_ds = train_val_split(dataset, split_ratio, seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    lr: float = 0.001,
    weight_decay: float = 1e-5,
    epochs: int = 20,
    early_stop_rounds: int = 5,
    model_save_path: str = "modelA_temp.pth"
) -> None:
    """
    Generic training loop for a regression model, with early stopping and MSE loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.3, patience=3, verbose=False)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    stop_count = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(features)
            preds = preds.squeeze(dim=-1)  # ensure shape matches
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        val_loss = evaluate_model(model, val_loader)
        scheduler.step(val_loss)

        print(f"[Epoch {epoch+1}/{epochs}] train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        # Early stopping check
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
    """
    Computes the MSE loss on a given data_loader (validation set).
    """
    device = next(model.parameters()).device
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for features, labels in data_loader:
            features, labels = features.to(device), labels.to(device)
            preds = model(features).squeeze(dim=-1)
            loss = criterion(preds, labels)
            running_loss += loss.item()
    return running_loss / len(data_loader)


def predict_model(model: nn.Module, data_loader: DataLoader) -> List[float]:
    """
    Returns predictions for all samples in data_loader.
    """
    device = next(model.parameters()).device
    preds_list = []
    model.eval()
    with torch.no_grad():
        for features, _ in data_loader:
            features = features.to(device)
            preds = model(features).squeeze(dim=-1)
            preds_list.extend(preds.cpu().tolist())
    return preds_list


def get_actuals(data_loader: DataLoader) -> List[float]:
    """
    Gets ground-truth labels (actuals) from data_loader in order.
    """
    actuals_list = []
    for _, labels in data_loader:
        actuals_list.extend(labels.tolist())
    return actuals_list


def compute_regression_metrics(y_true: List[float], y_pred: List[float]) -> dict:
    """
    Compute MSE and MAE for a regression problem and return as a dict.
    """
    mse = sum((yt - yp)**2 for yt, yp in zip(y_true, y_pred)) / len(y_true)
    mae = sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / len(y_true)
    return {"MSE": mse, "MAE": mae}


def create_double_lift_chart(
    predsA: List[float], 
    predsB: List[float], 
    actuals: List[float],
    n_deciles: int = 10
):
    """
    Create a double lift chart by decile. Steps:
      1) Combine predictions A, B, and Actual into a DataFrame.
      2) Sort by average of A & B or by either A or B, depending on preference.
      3) Split into deciles.
      4) Compute mean predsA, predsB, and actual per decile.
      5) Rebase actual to 1.0 and scale A/B by actual_mean to see ratio.
      6) Plot lines for A, B, and Actual=1.0 over the deciles.
    """
    df = pd.DataFrame({
        'predA': predsA,
        'predB': predsB,
        'actual': actuals
    })

    # Sort by the average of (predA + predB) / 2, or you could sort by predA only or predB only.
    df['avg_pred'] = (df['predA'] + df['predB']) / 2.0
    df.sort_values('avg_pred', inplace=True, ignore_index=True)

    # Create decile labels
    df['decile'] = pd.qcut(df['avg_pred'], n_deciles, labels=False, duplicates='drop')
    # If the dataset is small or has many ties, duplicates='drop' might create fewer than 10 bins

    # Group by decile and compute means
    grouped = df.groupby('decile').mean(numeric_only=True)

    # The grouped DataFrame has columns: predA, predB, actual, avg_pred
    # Rebase actual to 1.0 => we compute ratio_of_A = mean_predA / mean_actual, etc.
    # Then the "actual line" is always 1.0
    grouped['rebase_A'] = grouped['predA'] / grouped['actual']
    grouped['rebase_B'] = grouped['predB'] / grouped['actual']
    grouped['rebase_actual'] = 1.0  # The baseline

    # Plot the lines for rebase_A, rebase_B, rebase_actual
    fig = px.line(
        grouped[['rebase_A', 'rebase_B', 'rebase_actual']],
        labels={"value": "Prediction (rebased)", "index": "Decile"}
    )
    fig.update_layout(
        title="Double Lift Chart (Prediction vs. Actual Rebased=1.0)",
        xaxis_title="Decile",
        yaxis_title="Ratio to Actual",
    )
    fig.show()

# ------------------------ Main Script ------------------------

def main():
    """
    Main entry point.  
    1) Loads data for Model A (train_converted.pt), trains or loads the model, and obtains predictions on the val set.  
    2) Loads data for Model B (train.json), trains or loads the model, and obtains predictions on the val set.  
    3) Computes MSE / MAE for both, compares them.  
    4) Creates a double lift chart comparing both models to actuals.  
    """
    torch.manual_seed(0)  # for reproducibility if needed

    # ----------- MODEL A -----------
    print("=== Model A (EnsembleChessModel) ===")
    datasetA_path = "data/train_converted.pt"  # Adjust path as needed
    datasetA = ChessMoveDatasetA(datasetA_path)
    train_loaderA, val_loaderA = create_dataloaders(datasetA, batch_size=64, split_ratio=0.8, seed=1337)

    modelA = EnsembleChessModel()
    # Option A: train from scratch
    # train_model(modelA, train_loaderA, val_loaderA,
    #             epochs=15,
    #             model_save_path="modelA_temp.pth")  # Adjust as needed
    # Option B: or load a pre-trained model, e.g. modelA.load_state_dict(torch.load("model_gm_cnn.pth"))
    modelA.load_state_dict(torch.load("data/model_lichess_vastai.pth", map_location="cpu"))

    # load best checkpoint from training
    # modelA.load_state_dict(torch.load("modelA_temp.pth", map_location="cpu"))

    # Generate predictions on val set
    predsA = predict_model(modelA, val_loaderA)
    actualsA = get_actuals(val_loaderA)

    # ----------- MODEL B -----------
    print("=== Model B (ChessModelB) ===")
    datasetB_path = "data/train.json"  # Adjust path as needed
    datasetB = ChessMoveDatasetB(datasetB_path)
    train_loaderB, val_loaderB = create_dataloaders(datasetB, batch_size=64, split_ratio=0.8, seed=1337)

    modelB = ChessModelB()
    # Option A: train from scratch
    # train_model(modelB, train_loaderB, val_loaderB,
    #             epochs=15,
    #             model_save_path="modelB_temp.pth")  # Adjust as needed
    # Option B: or load a pre-trained model, e.g. modelB.load_state_dict(torch.load("model_fixval.pth"))
    modelB.load_state_dict(torch.load("data/model_fixval.pth"))

    # load best checkpoint from training
    # modelB.load_state_dict(torch.load("modelB_temp.pth", map_location="cpu"))

    predsB = predict_model(modelB, val_loaderB)
    actualsB = get_actuals(val_loaderB)

    # ----------- METRICS -----------
    # Because we know the labels align exactly, actualsA and actualsB should be the same.
    # For safety, we can verify the length and (optionally) they match within epsilon.
    if len(actualsA) != len(actualsB):
        raise ValueError("Mismatch in lengths of validation sets for Model A and B.")
    # We'll just use actualsA as the ground truth (they should be the same).
    actuals = actualsA

    # Compare MSE / MAE for each model
    metricsA = compute_regression_metrics(actuals, predsA)
    metricsB = compute_regression_metrics(actuals, predsB)

    print(f"Model A => MSE: {metricsA['MSE']:.6f}, MAE: {metricsA['MAE']:.6f}")
    print(f"Model B => MSE: {metricsB['MSE']:.6f}, MAE: {metricsB['MAE']:.6f}")

    # ----------- DOUBLE LIFT CHART -----------
    create_double_lift_chart(predsA, predsB, actuals, n_deciles=10)

    print("Done. See plots for the double lift chart comparison.")

if __name__ == "__main__":
    main()
