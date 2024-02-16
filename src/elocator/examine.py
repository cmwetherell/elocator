import torch
from torch.utils.data import DataLoader
from model_build import ChessModel  # Import your ChessModel
from utils import fen_decode  # Import fen_decode
from model_build import ChessMoveDataset, create_dataloader, load_dataset  # Import dataset and dataloader functions

# torch set seed
torch.manual_seed(0)

batch_size = 64

# Load the model
model = ChessModel(780)
model.load_state_dict(torch.load("./data/model.pth"))
model.eval()

# Set the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)

train_dataset, val_dataset = load_dataset('./data/train.json')
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Function to get predictions and FEN strings
def get_predictions_and_fens(dataloader):
    predictions = []
    fens = []
    with torch.no_grad():
        for features, label in dataloader:
            features = features.to(device)
            outputs = model(features).squeeze(1).tolist()  # Squeeze to match label shape
            predictions.extend(outputs)
            fens.extend([fen_decode(f) for f in features.tolist()])  # Decode FEN strings
    return predictions, fens

# Get predictions and FEN strings
predictions, fens = get_predictions_and_fens(val_dataloader)

# Find samples with highest and lowest predictions
max_pred_idx = predictions.index(max(predictions))
min_pred_idx = predictions.index(min(predictions))
highest_pred_sample = fens[max_pred_idx], predictions[max_pred_idx]
lowest_pred_sample = fens[min_pred_idx], predictions[min_pred_idx]

print("Sample with Highest Prediction:", highest_pred_sample)
print("Sample with Lowest Prediction:", lowest_pred_sample)

# find the 5th highest

# Sort predictions and FEN strings by prediction
sorted_predictions, sorted_fens = zip(*sorted(zip(predictions, fens)))

# Get the 5th highest prediction
fifth_highest = sorted_fens[-5]
print("5th Highest Prediction:", fifth_highest, sorted_predictions[-5])
print("4th Highest Prediction:", sorted_fens[-4], sorted_predictions[-4])
print("3rd Highest Prediction:", sorted_fens[-3], sorted_predictions[-3])
print("2nd Highest Prediction:", sorted_fens[-2], sorted_predictions[-2])
print("Highest Prediction:", sorted_fens[-1], sorted_predictions[-1])

