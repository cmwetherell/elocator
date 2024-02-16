import torch
from torch.utils.data import DataLoader
import json
import chess.pgn
import chess.engine
# from pgn_process import process_game
# from examine import get_predictions_and_fens
from model import ChessModel, ChessMoveDataset, create_dataloader, load_dataset  # Import dataset and dataloader functions
from utils import fen_decode
import pandas as pd

# need to score the moves in a game, compare to model expectations. Then, for each player evalute their expected rating for the game.
# output = players Elo
# input = move accuracy relative to expectation.

def append_predictions(model_pth: str, train_file: str) -> None:
    '''Takes a training file and model weights and appends model predictions to training file.'''

    with open(train_file, 'r') as file:
        gameData: dict = json.load(file)

    # torch set seed
    torch.manual_seed(0)

    batch_size = 32
    fen_size = 780

    # Load the model
    model = ChessModel(fen_size)
    model.load_state_dict(torch.load(model_pth))
    model.eval()

    # Set the device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)

    scoring_dataset = ChessMoveDataset(gameData)
    scoring_dataloader = DataLoader(scoring_dataset, batch_size=batch_size, shuffle=False)

    preds, _ = make_predictions(scoring_dataloader, device, model)

    # append preds to data
    data = pd.DataFrame(gameData)
    data['ModelPredictions'] = preds

    data.to_pickle('./data/scored.pkl')


def make_predictions(dataloader, device, model):
    predictions = []
    fens = []
    with torch.no_grad():
        for features, label in dataloader:
            features = features.to(device)
            outputs = model(features).squeeze(1).tolist()  # Squeeze to match label shape
            predictions.extend(outputs)
            fens.extend([fen_decode(f) for f in features.tolist()])  # Decode FEN strings
    return predictions, fens

def main():
    model_pth = "./data/model.pth"
    train_file = "./data/train.json"
    append_predictions(model_pth=model_pth, train_file=train_file)

if __name__ == "__main__":
    main()










# def process_game_model(game: chess.pgn, model_pth: str = './data/model.pth') -> [()]:
#     '''Proceess a game for expected vs actual accuracy'''

#     engine = chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish")
#     gameData = []
#     process_game(game, engine, gameData, position_scores={})

#     # torch set seed
#     torch.manual_seed(0)

#     batch_size = 64

#     # Load the model
#     model = ChessModel()
#     model.load_state_dict(torch.load("./data/model.pth"))
#     model.eval()

#     # Set the device
#     device = "mps" if torch.backends.mps.is_available() else "cpu"
#     model.to(device)

#     eval_dataset = ChessMoveDataset(gameData)
#     eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

#     # Get predictions and FEN strings
#     predictions, fens = get_predictions_and_fens(eval_dataloader)
    
# new function to process gameData with a model that conjectures the players rating gicven the moves and accuracy and expected accuracy


# make new dataset of FEN encoding + model expected score vs actual score. can start with current dataset.
# step 1 is to make comlete dataset
# start modeling
# some type of cumulative score normalized by move. e.g., expectation is 1.0, score is actual, product of relativities, relativities scatter plot with Elos, fit a simple line
# try second: use rnn or transformer nn architecture to predict Elo given sequence of move accuracy relative to expectation


# load train.json and then append a column for model predictions.
