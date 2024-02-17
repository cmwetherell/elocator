'''Elocator API application file'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
base_dir = Path(__file__).resolve().parent

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
from model_build import ChessModel
from utils import fen_encoder, parse_pgn, analyze_positions
from api.data_models import ComplexityRequest, GameRequest

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading model...")
# Model setup
model = ChessModel(fen_size=780)

device = "cpu"
if torch.backends.mps.is_available():
    try:
        # Attempt to use MPS device
        torch.tensor([], device="mps")
        device = "mps"
    except RuntimeError:
        print("MPS device not recognized, defaulting to CPU")
model.to(device)

# Path to the model file
model_path = base_dir / "model/model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()



# Defining the decile-to-complexity-score mapping
percentile_ranges = {
    1: (0, 0.006848618667572737),
    2: (0.006848618667572737, 0.007860606908798218),
    3: (0.007860606908798218, 0.0093873867765069),
    4: (0.0093873867765069, 0.010885232314467431),
    5: (0.010885232314467431, 0.01191701553761959),
    6: (0.01191701553761959, 0.012793240323662757),
    7: (0.012793240323662757, 0.013946877606213093),
    8: (0.013946877606213093, 0.015834777429699905),
    9: (0.015834777429699905, 0.02067287489771843),
    10: (0.02067287489771843, 1)
}

def map_new_prediction_to_complexity(new_prediction, percentile_ranges):
    """
    Maps a new prediction value to a complexity level based on predefined percentile ranges.

    Parameters:
    - new_prediction: The prediction value to map.
    - percentile_ranges: A dictionary with complexity levels as keys and (lower_bound, upper_bound) tuples as values.

    Returns:
    - The complexity level (1-10) for the new prediction.
    """
    for level, (low, high) in percentile_ranges.items():
        if low <= new_prediction <= high:
            return level
    return None  # Optionally handle predictions outside the expected range

def get_complexity_score(fen: str) -> int:
    """
    Get the complexity score for a given FEN string.

    Parameters:
    - fen: The FEN string to evaluate.

    Returns:
    - The complexity score (1-10) for the FEN string.
    """
    # Convert FEN to tensor and add a batch dimension
    encoded_fen = fen_encoder(fen)
    feature_tensor = torch.tensor(encoded_fen, dtype=torch.float32).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        prediction = model(feature_tensor).squeeze().item()

    # Interpret the prediction to return a complexity score
    # This step is simplified; you'd adjust this based on your model's output and desired complexity interpretation
    complexity_score = map_new_prediction_to_complexity(prediction, percentile_ranges)

    return complexity_score

@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Elocator API!",
    }

@app.post("/complexity/")
def get_complexity(request: ComplexityRequest):
    '''Get the complexity score for a given FEN string.
    
    Parameters:
    - request: A request object containing the FEN string to evaluate.
    
    Returns:
    - A dictionary containing the complexity score for the given FEN string.
    '''
    response = {
        "complexity_score": get_complexity_score(request.fen)
    }
    return response

@app.post("/analyze-game/")
def analyze_game(request: GameRequest):
    '''Analyze a game for complexity scores and other metrics.

    Parameters:
    - request: A request object containing the PGN of the game to analyze.

    Returns:
    - A dictionary containing the complexity scores for the game's positions.
    '''
    # Parse the PGN and get the FEN strings
    headers, FENs = parse_pgn(request.pgn)
    complexities = [get_complexity_score(fen) for fen in FENs]
    position_eval = analyze_positions(FENs) # score is always from whites perspective

    game_headers = headers
    game_analysis = [{
        "fen": FENs,
        "complexity": complexities,
        "evaluation": position_eval
    } for FENs, complexities, position_eval in zip(FENs, complexities, position_eval)]

    response = {
        "gameHeaders": game_headers,
        "positionAnalysis": game_analysis
    }

    return response

if __name__ == "__main__":
    import uvicorn
    import argparse

    # Set up argparse
    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI application on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the FastAPI application on")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port)
