'module to define pytorch architecture, dataloader, and training loop'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.amp import autocast, GradScaler
import json
import os
import math
import chess
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils import fen_encoder
import numpy as np


# ---------------------------------------------------------------------------
# Dataset & DataLoader
# ---------------------------------------------------------------------------

class ChessMoveDataset(Dataset):
    def __init__(self, json_file):
        if isinstance(json_file, str):
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        elif isinstance(json_file, list):
            self.data = json_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = torch.tensor(item['FEN'], dtype=torch.float32)
        label = torch.tensor(item['Accuracy'] / 100.0, dtype=torch.float32)
        return features, label


def load_dataset(json_file, val_fraction=0.1, test_fraction=0.1):
    """Split into train / val / test."""
    dataset = ChessMoveDataset(json_file)
    n = len(dataset)
    n_test = int(test_fraction * n)
    n_val = int(val_fraction * n)
    n_train = n - n_val - n_test
    return random_split(dataset, [n_train, n_val, n_test])


def create_dataloader(json_file, batch_size=32, shuffle=True):
    dataset = ChessMoveDataset(json_file)
    print(f"Dataset length: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SqueezeExcitation(nn.Module):
    """Channel attention: learn which feature maps matter most."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        scale = x.mean(dim=(2, 3))          # global avg pool → (B, C)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


class ResidualBlock(nn.Module):
    """Pre-activation residual block with SE attention."""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.se = SqueezeExcitation(channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(x))
        out = self.conv1(out)
        out = F.silu(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.se(out)
        return out + residual


# ---------------------------------------------------------------------------
# Main model — CNN backbone + global-feature MLP head
# ---------------------------------------------------------------------------

class ChessModel(nn.Module):
    """
    Hybrid CNN for chess position complexity prediction.

    Architecture:
      1. Reshape 768-dim board → (12, 8, 8) and run through convolutional
         residual blocks with squeeze-and-excitation attention.
      2. Concatenate global-average-pooled CNN features with metadata
         (castling rights + en-passant = 12 dims).
      3. Feed through an MLP head to predict accuracy ∈ [0, 1].

    Accepts the same 780-dim input as the original model for full
    backward-compatibility with existing data pipelines and API code.
    """

    def __init__(self, fen_size=780, num_blocks=6, channels=128, head_dropout=0.3):
        super().__init__()
        self.fen_size = fen_size
        self.channels = channels

        # --- CNN stem ---
        # Input: 12 piece-type planes on 8×8 board
        self.stem = nn.Sequential(
            nn.Conv2d(12, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

        # --- Residual tower ---
        self.tower = nn.Sequential(
            *[ResidualBlock(channels, dropout=0.1) for _ in range(num_blocks)]
        )

        # --- Policy / value style head ---
        # After global average pooling we get `channels` features from the board.
        # We concatenate the 12 metadata features (8 EP + 4 castling).
        metadata_dim = fen_size - 768  # = 12
        mlp_input = channels + metadata_dim

        self.head = nn.Sequential(
            nn.Linear(mlp_input, 256),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(256, 64),
            nn.SiLU(inplace=True),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Split board planes from metadata
        board = x[:, :768].view(-1, 12, 8, 8)
        metadata = x[:, 768:]

        # CNN backbone
        out = self.stem(board)
        out = self.tower(out)

        # Global average pool → (B, channels)
        out = out.mean(dim=(2, 3))

        # Concat metadata and predict
        out = torch.cat([out, metadata], dim=1)
        out = self.head(out)

        # Clamp to [0, 1] with sigmoid
        return torch.sigmoid(out)


# Keep SimplifiedChessModel for reference / ablation
class SimplifiedChessModel(nn.Module):
    def __init__(self, fen_size=780):
        super().__init__()
        self.fc1 = nn.Linear(fen_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 8)
        self.fc5 = nn.Linear(8, 1)

        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.kaiming_normal_(fc.weight, nonlinearity='leaky_relu')

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = torch.sigmoid(self.fc5(x))
        return x


# ---------------------------------------------------------------------------
# Learning-rate schedule: linear warmup then cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        if step < self.warmup_steps:
            scale = step / max(1, self.warmup_steps)
        else:
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        return [max(self.min_lr, base_lr * scale) for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_and_validate(
    model,
    train_dataloader,
    val_dataloader,
    epochs=80,
    lr=3e-4,
    weight_decay=1e-2,
    early_stop_rounds=20,
    grad_clip=1.0,
    save_path="./data/model.pth",
):
    device = _get_device()
    print(f"Using device: {device}")
    model.to(device)

    # --- Optimizer: AdamW with proper weight decay ---
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Schedule: warmup + cosine ---
    steps_per_epoch = len(train_dataloader)
    total_steps = epochs * steps_per_epoch
    warmup_steps = min(3 * steps_per_epoch, total_steps // 10)  # ~3 epochs warmup
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    # --- Loss: Huber (smooth L1) — robust to outlier accuracy values ---
    criterion = nn.SmoothL1Loss()

    # --- Mixed-precision scaler (CUDA only) ---
    use_amp = device == "cuda"
    scaler = GradScaler(device) if use_amp else None

    best_val_loss = float('inf')
    stop_count = 0
    best_state = None
    global_step = 0

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        num_batches = 0

        for features, label in train_dataloader:
            features, label = features.to(device), label.to(device)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with autocast(device):
                    outputs = model(features).squeeze(1)
                    loss = criterion(outputs, label)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features).squeeze(1)
                loss = criterion(outputs, label)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            scheduler.step()
            global_step += 1
            running_loss += loss.item()
            num_batches += 1

        avg_train = running_loss / num_batches

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_batches = 0
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features).squeeze(1)
                val_loss += criterion(val_outputs, val_labels).item()
                val_batches += 1
        avg_val = val_loss / val_batches

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d}/{epochs}  "
              f"Train: {avg_train:.6f}  Val: {avg_val:.6f}  "
              f"LR: {current_lr:.2e}")

        # ---- Checkpointing (best model) ----
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            stop_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            stop_count += 1

        if stop_count >= early_stop_rounds:
            print(f"Early stopped after {epoch + 1} epochs (best val: {best_val_loss:.6f})")
            break

    # ---- Save best model ----
    if best_state is not None:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        torch.save(best_state, save_path)
        print(f"Saved best model (val loss {best_val_loss:.6f}) to {save_path}")
    else:
        torch.save(model.state_dict(), save_path)
        print(f"Saved model to {save_path}")


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(model, fen):
    model.load_state_dict(torch.load("./data/model.pth", weights_only=True))
    device = _get_device()
    model.to(device)
    model.eval()
    fen_tensor = torch.tensor(fen_encoder(fen), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(fen_tensor)
    return prediction.item()


def get_predictions(model, dataloader, device):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            predictions.extend(outputs.view(-1).tolist())
            actuals.extend(labels.view(-1).tolist())
    return predictions, actuals


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    batch_size = 256
    fen_size = 780
    num_workers = 4

    device = _get_device()
    print(f"Using device: {device}")

    train_dataset, val_dataset, test_dataset = load_dataset('./data/train.json')
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device != "cpu"), persistent_workers=True,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"), persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device != "cpu"), persistent_workers=True,
    )

    model = ChessModel(fen_size)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    train_and_validate(model, train_dataloader, val_dataloader)

    # ---- Evaluate on held-out test set ----
    model.load_state_dict(torch.load("./data/model.pth", weights_only=True))
    model.to(device)
    test_predictions, test_actuals = get_predictions(model, test_dataloader, device)
    test_df = pd.DataFrame({'Actual': test_actuals, 'Predicted': test_predictions})
    mse = ((test_df['Actual'] - test_df['Predicted']) ** 2).mean()
    mae = (test_df['Actual'] - test_df['Predicted']).abs().mean()
    print(f"\nTest MSE: {mse:.6f}  |  Test MAE: {mae:.6f}")

    # ---- Lift chart on validation set ----
    val_predictions, val_actuals = get_predictions(model, val_dataloader, device)
    val_df = pd.DataFrame({'Actual': val_actuals, 'Predicted': val_predictions})

    try:
        val_df['Decile'] = pd.qcut(val_df['Predicted'], 10, labels=False, duplicates='drop') + 1
    except ValueError as e:
        print("Error in creating deciles: ", e)

    val_decile_means = val_df.groupby('Decile').mean().round(4)

    fig = px.line(val_decile_means, y=['Actual', 'Predicted'])
    fig.update_layout(
        title="Actual vs Predicted by Complexity Score",
        xaxis_title="Complexity Score",
        yaxis_title="Expected Win % Reduction",
        legend_title="Type",
        barmode='group'
    )
    fig.update_layout({
        'plot_bgcolor': 'white',
        'paper_bgcolor': 'white',
        'xaxis': {
            'showgrid': False, 'zeroline': False,
            'ticks': 'outside', 'tickcolor': 'black',
        },
        'yaxis': {
            'showgrid': False, 'zeroline': False,
            'ticks': 'outside', 'tickcolor': 'black',
            'tickformat': '.1%',
        },
        'margin': {'l': 40, 'r': 20, 't': 20, 'b': 30},
    })
    fig.update_traces(line=dict(width=4))
    fig.show()
    div = pio.to_html(fig, full_html=False)
    with open('div.html', 'w') as f:
        f.write(div)

    # Diagnostic predictions
    print(f"\nStarting position: {predict(model, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):.5f}")
    print(f"Complex position:  {predict(model, '4kb1r/1p1n1ppp/p3b3/4p3/q3p3/P1P1B1QP/3NKPP1/3R1B1R w k - 2 19'):.5f}")
    print(f"Italian Game:      {predict(model, 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3'):.5f}")


if __name__ == "__main__":
    main()
