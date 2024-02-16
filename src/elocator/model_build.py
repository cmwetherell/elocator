'module to define pytorch architecture, dataloader, etc'

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import json
import os
import chess
import pandas as pd
import plotly.express as px
import plotly.io as pio
from utils import fen_encoder
import numpy as np

class ChessMoveDataset(Dataset):
    def __init__(self, json_file):
        if type(json_file) == str:
            with open(json_file, 'r') as file:
                self.data = json.load(file)
        elif type(json_file) == list:
            self.data = json_file

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # Convert the FEN list to a tensor with dtype torch.uint8 or torch.bool
        features = torch.tensor(item['FEN'], dtype=torch.float32)  # or dtype=torch.uint8

        # The target variable is 'Accuracy'
        label = torch.tensor(item['Accuracy'], dtype=torch.float32)
        # divide label by 100 to get a number between 0 and 1
        label /= 100

        return features, label

def load_dataset(json_file):
    dataset = ChessMoveDataset(json_file)
    train_size = int(0.8 * len(dataset))  # 80% of the data for training
    val_size = len(dataset) - train_size  # 20% for validation
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

# Function to create DataLoader with an option to shuffle or not
def create_dataloader(json_file, batch_size=32, shuffle=True):
    dataset = ChessMoveDataset(json_file)
    print(f"Dataset length: {len(dataset)}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

# # Create a non-shuffled DataLoader for lift chart analysis
# train_dataloader_for_analysis = create_dataloader('./data/train.json', shuffle=False)

# # Example Usage
# train_dataloader = create_dataloader('./data/train.json')


# Define the model
# Use a simple nn to go from 780 to the accuracy prediction in a few steps, accuracy is a number between 0 and 1, but it is not a binary classification poroblem.
    
    # go from fen_size to 1024 to 512 to 128 to 16 to 1
    
# Define the enhanced model
class ChessModel(nn.Module):
    def __init__(self, fen_size):
        super().__init__()
        # Increased model complexity
        self.fc1 = nn.Linear(fen_size, 4096)
        self.fc2 = nn.Linear(4096, 2056)
        self.fc3 = nn.Linear(2056, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)  # New layer from 64 to 8
        self.fc7 = nn.Linear(8, 1)   # Adjusted final layer from 8 to 1

        # Initialize weights using Kaiming (He) initialization
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc7.weight)  # Assuming linear output; adjust if different activation is used

        # Add batch normalization and dropout layers
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2056)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

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
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)  # Use Leaky ReLU for the new layer as well
        x = torch.sigmoid(self.fc7(x))  # Sigmoid activation for the final layer, assuming a binary classification or similar task
        return x

class SimplifiedChessModel(nn.Module):
    def __init__(self, fen_size):
        super().__init__()
        # Updated model with an additional layer
        self.fc1 = nn.Linear(fen_size, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 8)
        self.fc4 = nn.Linear(8, 8)   # New layer with 8 units
        self.fc5 = nn.Linear(8, 1)   # Final layer to output 1 unit

        # Initialize weights using Kaiming (He) initialization for all layers
        for fc in [self.fc1, self.fc2, self.fc3, self.fc4, self.fc5]:
            nn.init.kaiming_normal_(fc.weight, nonlinearity='leaky_relu')

        # Optional: Add batch normalization and dropout to the updated model
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(0.5)  # Adjust dropout rate as needed

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01)
        x = self.dropout(x)
        x = F.leaky_relu(self.fc4(x), negative_slope=0.01)
        x = torch.sigmoid(self.fc5(x))  # Sigmoid for the final output layer
        return x

# class ChessModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Convolutional layers for the chessboard
#         self.conv1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

#         # Pooling layer
#         self.pool = nn.MaxPool2d(2, 2)

#         # Fully connected layers
#         # Adjust the input size of the first FC layer to include original features
#         self.fc1 = nn.Linear(512 + 780, 1024)
#         self.fc2 = nn.Linear(1024, 128)
#         self.fc3 = nn.Linear(128, 1)  # Output layer

#         # Batch normalization
#         self.bn1 = nn.BatchNorm2d(64)
#         self.bn2 = nn.BatchNorm2d(128)
#         self.bn3 = nn.BatchNorm1d(1024)
#         self.bn4 = nn.BatchNorm1d(128)

#         # Dropout
#         self.dropout = nn.Dropout(0.5)

#         # Initialize weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             elif isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)

#     def forward(self, x):
#         # Process chessboard through convolutional layers
#         chessboard = x[:, :768].view(-1, 12, 8, 8)
#         x_conv = self.pool(F.relu(self.bn1(self.conv1(chessboard))))
#         x_conv = self.pool(F.relu(self.bn2(self.conv2(x_conv))))
#         x_conv = x_conv.view(-1, 512)  # Correctly flattened

#         # Concatenate with the original features
#         x_original = x.view(x.size(0), -1)
#         x_combined = torch.cat((x_conv, x_original), dim=1)

#         # Fully connected layers
#         x = F.relu(self.bn3(self.fc1(x_combined)))
#         x = self.dropout(x)
#         x = F.relu(self.bn4(self.fc2(x)))
#         x = self.dropout(x)
#         x = torch.sigmoid(self.fc3(x)) # Sigmoid activation for the output layer

#         return x
    

# write training loop function
    
def train_and_validate(model, train_dataloader, val_dataloader, epochs=40, lr=0.001, weight_decay=1e-5, early_stop_rounds=15):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=5, verbose=True)
    criterion = nn.MSELoss()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    stop_count = 0
    bst_val = 999

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (features, label) in enumerate(train_dataloader):
            features, label = features.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            outputs = outputs.squeeze(1)  # Squeeze the output to match the target's shape
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation phase at the end of each epoch
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_features, val_labels in val_dataloader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features)
                val_outputs = val_outputs.squeeze(1)
                val_loss += criterion(val_outputs, val_labels).item()
        val_loss /= len(val_dataloader)

        # Print training and validation loss
        print(f"Epoch {epoch+1}, Train Loss: {running_loss/len(train_dataloader):.6f}, Validation Loss: {val_loss:.6f}")

        # Step the scheduler
        scheduler.step(val_loss)

        # early stopping
        if val_loss < bst_val:
            bst_val = val_loss
            stop_count = 0
        else:
            stop_count += 1
        
        if stop_count < early_stop_rounds:
            continue
        elif stop_count > early_stop_rounds:
            raise Exception("Error, should have stopped but didn't")
        else:
            print(f"Early stopped after {epoch + 1}")
            break
            

    # Save model
    torch.save(model.state_dict(), "./data/model.pth")
    print("Finished Training, model saved.")

# predict accuracy loss given a FEN

def predict(model, fen):
    # Load model
    model.load_state_dict(torch.load("./data/model.pth"))
    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.eval()

    # Convert FEN to tensor and add a batch dimension
    fen_tensor = torch.tensor(fen_encoder(fen), dtype=torch.float32).unsqueeze(0).to(device)

    # Make prediction
    prediction = model(fen_tensor)
    return prediction.item()

def get_predictions(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
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

def main():
    # torch set seed
    torch.manual_seed(0)

    batch_size = 32
    fen_size = 780

    # Set the device      
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    # device = "cpu"
    print(f"Using device: {device}")

    train_dataset, val_dataset = load_dataset('./data/train.json')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # train the model
    model = ChessModel(fen_size)
    # train_and_validate(model, train_dataloader, val_dataloader)

    model.load_state_dict(torch.load("./data/model.pth"))
    model.to(device)

    # Get predictions for the validation dataset
    val_predictions, val_actuals = get_predictions(model, val_dataloader, device)

    # Create a DataFrame with actual and predicted values from validation data
    val_df = pd.DataFrame({'Actual': val_actuals, 'Predicted': val_predictions})

    # Calculate deciles
    try:
        val_df['Decile'] = pd.qcut(val_df['Predicted'], 10, labels=False, duplicates='drop') + 1
    except ValueError as e:
        print("Error in creating deciles: ", e)
        # Handle the error or perform alternative analysis

    # Calculate mean actual and predicted values for each decile
    val_decile_means = val_df.groupby('Decile').mean()

    # round actual and predicted columns to 4 decimals
    val_decile_means = val_decile_means.round(4)


    # Plotting the lift chart for validation data
    fig = px.line(val_decile_means, y=['Actual', 'Predicted'])
    fig.update_layout(
        title="Actual vs Predicted by Complexity Score",
        xaxis_title="Complexity Score",
        yaxis_title="Expected Win % Reduction",
        legend_title="Type",
        barmode='group'
    )
    fig.update_layout({
    'plot_bgcolor': 'white', # Set the background color to white
    'paper_bgcolor': 'white', # Set the surrounding color to white
    'xaxis': {
        'showgrid': False, # Hide the x-axis gridlines
        'zeroline': False, # Hide the x-axis zero line
        'ticks': 'outside', # Position ticks outside the plot
        'tickcolor': 'black', # Set tick color to black for visibility
    },
    'yaxis': {
        'showgrid': False, # Show y-axis gridlines for reference (optional)
        'zeroline': False, # Hide the y-axis zero line
        'ticks': 'outside', # Position ticks outside the plot
        'tickcolor': 'black', # Set tick color to black for visibility
        'tickformat': '.1%',
    },
    'margin': {'l': 40, 'r': 20, 't': 20, 'b': 30}, # Adjust margins to taste
})
    fig.update_traces(line=dict(width=4))
    fig.show()
    div = pio.to_html(fig, full_html=False)
    # save div object to text file
    with open('div.html', 'w') as f:
        f.write(div)

    # dianostic stuff

    print(f"{predict(model, 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq -'):.5f}")
    print(f"{predict(model, '4kb1r/1p1n1ppp/p3b3/4p3/q3p3/P1P1B1QP/3NKPP1/3R1B1R w k - 2 19'):.5f}")
    print(f"{predict(model, 'r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 1 3'):.5f}")


# def main():
#     torch.manual_seed(0)
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     print(f"Using device: {device}")

#     _, val_dataset = load_dataset('./data/train.json')  # Assuming this function is defined elsewhere
#     val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#     model = ChessModel(780)  # Initialize your model here
#     model.load_state_dict(torch.load("./data/model.pth", map_location=device))
#     model.to(device)
#     model.eval()

#     predictions, actuals = get_predictions(model, val_dataloader, device)  # Assuming this function is defined elsewhere

#     val_df = pd.DataFrame(predictions, columns=['Predicted'])

#     # Calculate deciles
#     val_df['Decile'] = pd.qcut(val_df['Predicted'], 10, labels=False, duplicates='drop') + 1

#     # Calculate percentile ranges for mapping
#     percentile_ranges = {i: (val_df['Predicted'].quantile((i-1)/10), val_df['Predicted'].quantile(i/10)) for i in range(1, 11)}

#     complexity_mapping = {i: "Complexity Level " + str(i) for i in range(1, 11)}
#     print("Complexity Mapping Dictionary:", complexity_mapping)

#     print(val_df.groupby('Decile')['Predicted'].mean())

#     val_decile_means = val_df.groupby('Decile')['Predicted'].mean()
#     fig = px.bar(val_decile_means, y='Predicted', labels={'value': 'Predicted Complexity', 'Decile': 'Decile'})
#     fig.update_layout(title="Predicted Complexity by Decile", xaxis_title="Decile", yaxis_title="Predicted Complexity")
#     fig.show()

#     return complexity_mapping, percentile_ranges

# def map_new_prediction_to_complexity(new_prediction, percentile_ranges):
#     for level, (low, high) in percentile_ranges.items():
#         if low <= new_prediction <= high:
#             return level
#     return None



if __name__ == "__main__":
    # c, p = main()
    # print(p)
    main()
