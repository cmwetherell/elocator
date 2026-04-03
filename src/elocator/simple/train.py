import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from elocator.utils import fen_to_tensor
import pandas as pd
import plotly.express as px

# ------------------------ Data Preparation ------------------------

class ChessMoveDataset(Dataset):
    """
    Dataset for loading chess move data stored in a PyTorch tensor format.
    """
    def __init__(self, tensor_file_path):
        """
        Initialize the dataset.

        :param tensor_file_path: Path to the saved tensor file.
        """
        data = torch.load(tensor_file_path)
        self.features = data['FENs']
        self.labels = torch.tensor([item['Accuracy'] for item in data['Metadata']], dtype=torch.float32) / 100

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class ChessDataLoader:
    """
    Handles loading and splitting the dataset into train and validation sets with DataLoaders.
    """
    def __init__(self, dataset, split_ratio=0.8, batch_size=32):
        """
        Initialize the DataLoader.

        :param dataset: ChessMoveDataset instance
        :param split_ratio: Ratio of train to validation split
        :param batch_size: Batch size for DataLoaders
        """
        train_size = int(split_ratio * len(dataset))
        val_size = len(dataset) - train_size
        
        generator = torch.Generator().manual_seed(1337)
        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        self.batch_size = batch_size

    def get_loaders(self):
        """
        Return train and validation DataLoaders.
        """
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False)
        return train_loader, val_loader

# ------------------------ Model Definitions ------------------------

def init_kaiming(m):
    """
    Utility function to apply Kaiming normal initialization to Conv2d or Linear layers,
    and set BatchNorm2d parameters to sensible defaults.
    """
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, a=0.01, mode='fan_out', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

class SubNetworkA(nn.Module):
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

        # Apply Kaiming initialization
        self.apply(init_kaiming)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dropout2(x)

        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.01)

        x = self.gap(x).view(x.size(0), -1)
        return x


class SubNetworkB(nn.Module):
    def __init__(self, in_channels=18):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.3)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc = nn.Linear(32 * 8 * 8, 64)

        # Apply Kaiming initialization
        self.apply(init_kaiming)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.01)
        x = self.dropout1(x)

        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.01)
        x = self.dropout2(x)

        x = x.view(x.size(0), -1)
        return self.fc(x)


class EnsembleChessModel(nn.Module):
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

        # Apply Kaiming initialization
        self.apply(init_kaiming)

    def forward(self, x):
        embA = self.subA(x)
        embB = self.subB(x)
        return self.fc(torch.cat([embA, embB], dim=-1))

# ------------------------ Training Loop ------------------------

class Trainer:
    def __init__(self, model, train_loader, val_loader, lr=0.001, weight_decay=1e-5):
        """
        Initialize the Trainer.

        :param model: Model to train
        :param train_loader: Training DataLoader
        :param val_loader: Validation DataLoader
        :param lr: Learning rate
        :param weight_decay: Weight decay for optimizer
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=5)
        self.criterion = nn.MSELoss()

    def train(self, epochs=100, early_stop_rounds=10):
        stop_count, best_val_loss = 0, float('inf')

        for epoch in range(epochs):
            self.model.train()
            running_loss = 0.0

            for features, labels in self.train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                predictions = self.model(features)
                loss = self.criterion(predictions.squeeze(), labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()

            val_loss = self.validate()
            print(f"Epoch {epoch+1}: Train Loss = {running_loss/len(self.train_loader):.6f}, Validation Loss = {val_loss:.6f}")

            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss, stop_count = val_loss, 0
                torch.save(self.model.state_dict(), "./data/model_lichess.pth")
            else:
                stop_count += 1

            if stop_count >= early_stop_rounds:
                print("Early stopping triggered.")
                break

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features)
                val_loss += self.criterion(predictions.squeeze(), labels)
        return val_loss / len(self.val_loader)

    def evaluate(self):
        self.model.eval()
        val_predictions, val_actuals = [], []
        with torch.no_grad():
            for features, labels in self.val_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                predictions = self.model(features).squeeze()
                val_predictions.extend(predictions.cpu().numpy())
                val_actuals.extend(labels.cpu().numpy())

        val_df = pd.DataFrame({'Actual': val_actuals, 'Predicted': val_predictions})

        # print unique values counts of predicted values
        print(val_df['Predicted'].value_counts())

        # create scatter plot of actual vs predicted with a line of best fit
        fig = px.scatter(val_df, x='Actual', y='Predicted', trendline='ols')
        fig.update_layout(
            title="Actual vs Predicted Accuracy Loss",
            xaxis_title="Actual Accuracy Loss",
            yaxis_title="Predicted Accuracy Loss",
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin={'l': 40, 'r': 20, 't': 20, 'b': 30}
        )
        fig.show()

        try:
            val_df['Decile'] = pd.qcut(val_df['Predicted'], 10, labels=False, duplicates='drop') + 1
        except ValueError as e:
            print("Error in creating deciles: ", e)

        val_decile_means = val_df.groupby('Decile').mean().round(4)
        print(val_decile_means)

        fig = px.line(val_decile_means, y=['Actual', 'Predicted'])
        fig.update_layout(
            title="Actual vs Predicted by Complexity Score",
            xaxis_title="Complexity Score",
            yaxis_title="Expected Win % Reduction",
            legend_title="Type",
            barmode='group',
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin={'l': 40, 'r': 20, 't': 20, 'b': 30}
        )
        fig.update_traces(line=dict(width=4))
        fig.show()

# ------------------------ Main ------------------------

def main():
    dataset_path = 'data/lichess_gameData_converted.pt'
    dataset = ChessMoveDataset(dataset_path)
    data_loader = ChessDataLoader(dataset)
    train_loader, val_loader = data_loader.get_loaders()

    model = EnsembleChessModel()
    trainer = Trainer(model, train_loader, val_loader)
    trainer.train()

    model.load_state_dict(torch.load("./data/model_lichess.pth"))
    # note: best model is "model_gm_cnn.pth"
    trainer.evaluate()

    test_fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -",
        "4kb1r/1p1n1ppp/p3b3/4p3/q3p3/P1P1B1QP/3NKPP1/3R1B1R w k - 2 19",
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3",
        "4rk2/ppp1qppp/3p2R1/8/4P3/2Q1R2P/PPP2PP1/6K1 b - - 0 1",
        "2kr3r/ppqb4/3p1b1p/2pPnpp1/NPP1p1nP/6PB/PB2PPN1/2RQ1RK1 w - - 0 1"
    ]

    for fen in test_fens:
        tensor = fen_to_tensor(fen).unsqueeze(0).to(trainer.device)
        prediction = model(tensor).item()
        print(f"FEN: {fen}\nPredicted Accuracy Loss: {prediction:.5f}\n")


if __name__ == "__main__":
    main()
