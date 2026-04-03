import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import pickle
from utils import fen_to_tensor, fen_encoder, fen_decode
import random
import pandas as pd
import plotly.express as px
import plotly.io as pio

# ------------------------ Data Preparation ------------------------

class ChessMoveDataset(Dataset):
    """Dataset for loading chess move data."""
    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'r') as file:
                self.data = json.load(file)
        elif isinstance(data, list):
            self.data = data
        else:
            raise ValueError("Unsupported data format")
        print(f"Dataset initialized with {len(self.data)} items.")  # Debug print


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        decoded_fen = fen_decode(item['FEN'])
        features = fen_to_tensor(decoded_fen)
        label = torch.tensor(item['Accuracy'], dtype=torch.float32) / 100
        return features, label


def load_dataset(data, split_ratio=0.8):
    """Load and split the dataset into training and validation sets."""
    dataset = ChessMoveDataset(data)
    dataset = random.sample(dataset.data, 25000)  # Take a subset of the dataset for testing
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


def create_dataloader(dataset, batch_size=32, shuffle=True):
    """Create a DataLoader from the dataset."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# ------------------------ Model Definitions ------------------------

class SubNetworkA(nn.Module):
    """Deeper CNN for feature extraction."""
    def __init__(self, in_channels=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.gap(x).view(x.size(0), -1)
        return x


class SimpleAttentionBlock(nn.Module):
    """Attention mechanism for SubNetworkB."""
    def __init__(self, embed_dim=32):
        super().__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attn_scores = torch.matmul(Q, K.transpose(-1, -2)) / (Q.size(-1) ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)
        return torch.matmul(attn_weights, V)


class SubNetworkB(nn.Module):
    """Smaller CNN with attention block."""
    def __init__(self, in_channels=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.attention = SimpleAttentionBlock(embed_dim=32)
        self.fc = nn.Linear(64 * 32, 64)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(x.size(0), 32, -1).transpose(1, 2)
        x = self.attention(x).view(x.size(0), -1)
        return self.fc(x)


class EnsembleChessModel(nn.Module):
    """Ensemble model combining SubNetworkA and SubNetworkB."""
    def __init__(self):
        super().__init__()
        self.subA = SubNetworkA()
        self.subB = SubNetworkB()
        self.fc = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        embA = self.subA(x)
        embB = self.subB(x)
        return self.fc(torch.cat([embA, embB], dim=-1))

# ------------------------ Training Loop ------------------------

def train_and_validate(model, train_dataloader, val_dataloader, epochs=100, lr=0.01, weight_decay=1e-5, early_stop_rounds=10):
    """Training loop with early stopping."""
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    criterion = nn.MSELoss()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    stop_count, best_val_loss = 0, float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_dataloader:
            print(batch)  # This will print the structure of the batch
            break  # Check only one batch to avoid excessive output

        for features, labels in train_dataloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions.squeeze(), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in val_dataloader:
                features, labels = features.to(device), labels.to(device)
                predictions = model(features)
                val_loss += criterion(predictions.squeeze(), labels).item()
        val_loss /= len(val_dataloader)

        print(f"Epoch {epoch+1}: Train Loss = {running_loss/len(train_dataloader):.6f}, Validation Loss = {val_loss:.6f}")

        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, stop_count = val_loss, 0
        else:
            stop_count += 1

        if stop_count >= early_stop_rounds:
            print("Early stopping triggered.")
            break

    torch.save(model.state_dict(), "./data/model_lichess.pth")
    print("Model saved.")

# ------------------------ Main ------------------------

def main():
    """Main function to train the model and perform diagnostics."""
    torch.manual_seed(0)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load dataset
    data_path = 'data/lichess_gameData.pkl'
    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    # Split and create dataloaders
    train_dataset, val_dataset = load_dataset(data)
    train_dataloader = create_dataloader(train_dataset, batch_size=4)
    val_dataloader = create_dataloader(val_dataset, batch_size=4, shuffle=False)

    # Initialize model
    model = EnsembleChessModel()
    train_and_validate(model, train_dataloader, val_dataloader)

    # Reload best model for diagnostics
    model.load_state_dict(torch.load("./data/model_lichess.pth"))
    model.to(device)

    # Diagnostic testing with specific FEN strings
    model.eval()
    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "4kb1r/1p1n1ppp/p3b3/4p3/q3p3/P1P1B1QP/3NKPP1/3R1B1R w k - 2 19",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3"
    ]

    print("\nDiagnostic Predictions:")
    for fen in test_fens:
        fen_tensor = fen_to_tensor(fen)
        fen_tensor = fen_tensor.unsqueeze(0).to(device)
        prediction = model(fen_tensor).item()
        print(f"FEN: {fen}\nPredicted Accuracy Loss: {prediction:.5f}\n")

if __name__ == "__main__":
    main()
