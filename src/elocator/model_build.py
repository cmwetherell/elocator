"""Dataset, data loading, and training loop for Elocator CNN model."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import pandas as pd
import plotly.express as px
import plotly.io as pio

from utils import fen_to_tensor
from model_cnn import ChessCNNModel


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def tweedie_loss(y_pred, y_true, p=1.5):
    """Tweedie deviance loss for zero-inflated right-skewed data.

    Args:
        y_pred: Positive predictions (from Softplus output).
        y_true: Target values >= 0.
        p: Power parameter. 1.5 is a good default for zero-inflated continuous data.
    """
    y_pred = y_pred.clamp(min=1e-6)
    return torch.mean(
        -y_true * y_pred.pow(1 - p) / (1 - p) + y_pred.pow(2 - p) / (2 - p)
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ChessMoveDataset(Dataset):
    """Dataset that loads JSONL with raw FEN strings and encodes on-the-fly."""

    def __init__(self, data_source):
        """
        Args:
            data_source: Path to a .jsonl file, or a list of dicts.
                Each record must have 'fen' (str) and 'accuracy' (float 0-100).
        """
        if isinstance(data_source, str):
            self.data = []
            with open(data_source, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        elif isinstance(data_source, list):
            self.data = data_source
        else:
            raise ValueError("data_source must be a file path or list of dicts")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        features = fen_to_tensor(item['fen'])  # (18, 8, 8)
        label = torch.tensor(item['accuracy'], dtype=torch.float32)
        return features, label


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(data_source, train_ratio=0.8, val_ratio=0.1, seed=1337):
    """Load dataset and split into train/val/test (80/10/10)."""
    dataset = ChessMoveDataset(data_source)
    n = len(dataset)
    train_size = int(train_ratio * n)
    val_size = int(val_ratio * n)
    test_size = n - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    return train_dataset, val_dataset, test_dataset


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def create_loaders(train_dataset, val_dataset, test_dataset,
                   batch_size=256, num_workers=4):
    """Create DataLoaders with pinned memory and persistent workers."""
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(train_dataset, shuffle=True, **common)
    val_loader = DataLoader(val_dataset, shuffle=False, **common)
    test_loader = DataLoader(test_dataset, shuffle=False, **common)
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_and_validate(
    model,
    train_loader,
    val_loader,
    test_loader,
    max_epochs=200,
    lr=1e-3,
    weight_decay=1e-2,
    warmup_epochs=5,
    patience=20,
    tweedie_p=1.5,
    save_path="./data/model_best.pth",
):
    device = get_device()
    print(f"Using device: {device}")
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Linear warmup + cosine decay
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_epochs - warmup_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    # AMP (CUDA only; inactive on MPS)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # --- Training ---
        model.train()
        running_loss = 0.0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            if use_amp:
                with torch.autocast(device_type="cuda"):
                    outputs = model(features).squeeze(1)
                    loss = tweedie_loss(outputs, labels, p=tweedie_p)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(features).squeeze(1)
                loss = tweedie_loss(outputs, labels, p=tweedie_p)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        train_loss = running_loss / len(train_loader)

        # --- Validation ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features).squeeze(1)
                val_loss += tweedie_loss(outputs, labels, p=tweedie_p).item()
        val_loss /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {current_lr:.2e}")

        # --- Checkpointing ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, save_path)
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}. Best epoch: {best_epoch + 1}")
            break

    # --- Test evaluation ---
    print(f"\nLoading best model from epoch {best_epoch + 1}...")
    checkpoint = torch.load(save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_loss = 0.0
    test_preds = []
    test_actuals = []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features).squeeze(1)
            test_loss += tweedie_loss(outputs, labels, p=tweedie_p).item()
            test_preds.extend(outputs.cpu().tolist())
            test_actuals.extend(labels.cpu().tolist())

    test_loss /= len(test_loader)
    mae = sum(abs(p - a) for p, a in zip(test_preds, test_actuals)) / len(test_preds)
    print(f"Test Loss: {test_loss:.6f} | Test MAE: {mae:.6f}")

    print("Finished Training.")
    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(model, fen, model_path="./data/model_best.pth"):
    """Predict accuracy loss for a single FEN position."""
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    feature_tensor = fen_to_tensor(fen).unsqueeze(0).to(device)
    with torch.no_grad():
        prediction = model(feature_tensor)
    return prediction.item()


def get_predictions(model, dataloader, device):
    """Get predictions and actuals from a dataloader."""
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            outputs = model(features).squeeze(1)
            predictions.extend(outputs.cpu().tolist())
            actuals.extend(labels.tolist())
    return predictions, actuals


# ---------------------------------------------------------------------------
# Evaluation / visualization
# ---------------------------------------------------------------------------

def evaluate_and_plot(model, val_loader, device):
    """Generate lift chart comparing actual vs predicted by decile."""
    predictions, actuals = get_predictions(model, val_loader, device)
    val_df = pd.DataFrame({'Actual': actuals, 'Predicted': predictions})

    try:
        val_df['Decile'] = pd.qcut(val_df['Predicted'], 10, labels=False, duplicates='drop') + 1
    except ValueError as e:
        print("Error in creating deciles:", e)
        return

    val_decile_means = val_df.groupby('Decile').mean().round(4)
    print(val_decile_means)

    fig = px.line(val_decile_means, y=['Actual', 'Predicted'])
    fig.update_layout(
        title="Actual vs Predicted by Complexity Score",
        xaxis_title="Complexity Score",
        yaxis_title="Expected Win % Reduction",
        legend_title="Type",
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin={'l': 40, 'r': 20, 't': 40, 'b': 30},
    )
    fig.update_traces(line=dict(width=4))
    fig.show()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    torch.manual_seed(0)

    data_path = './data/train.jsonl'
    train_dataset, val_dataset, test_dataset = load_dataset(data_path)
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")

    train_loader, val_loader, test_loader = create_loaders(
        train_dataset, val_dataset, test_dataset
    )

    model = ChessCNNModel()
    model = train_and_validate(model, train_loader, val_loader, test_loader)

    device = get_device()
    model.to(device)
    evaluate_and_plot(model, val_loader, device)

    # Smoke test predictions
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "4kb1r/1p1n1ppp/p3b3/4p3/q3p3/P1P1B1QP/3NKPP1/3R1B1R w k - 2 19",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3",
    ]
    for fen in test_fens:
        pred = predict(model, fen)
        print(f"FEN: {fen[:60]}...  Predicted: {pred:.5f}")


if __name__ == "__main__":
    main()
