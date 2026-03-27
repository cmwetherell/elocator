'''Elocator API application file — Ensemble model (CNN + MLP rank average)'''

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
base_dir = Path(__file__).resolve().parent

from fastapi import FastAPI, HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model_cnn import ChessCNNModel
from utils import fen_to_tensor, fen_encoder, parse_pgn, analyze_positions
from api.data_models import ComplexityRequest, GameRequest
import logging
import time
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ElocatorAPI")

app = FastAPI()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_body = await request.body()
        try:
            request_body_json = json.loads(request_body.decode("utf-8"))
        except json.JSONDecodeError:
            request_body_json = "Unable to decode JSON"

        logger.info(f"Request path: {request.url.path}, Method: {request.method}, Body: {request_body_json}")
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(f"Response status: {response.status_code}, Time: {process_time}s")
        return response

app.add_middleware(LoggingMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Old MLP architecture (needed for ensemble)
# ---------------------------------------------------------------------------
class ChessModel(nn.Module):
    def __init__(self, fen_size):
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
        return torch.sigmoid(self.fc7(x))

# ---------------------------------------------------------------------------
# Load ensemble models
# ---------------------------------------------------------------------------
print("Loading ensemble models...")
device = "cpu"
if torch.backends.mps.is_available():
    try:
        torch.tensor([], device="mps")
        device = "mps"
    except RuntimeError:
        print("MPS device not recognized, defaulting to CPU")

# CNN: SE-ResNet with stochastic depth (Tweedie-trained, 1.18M positions)
cnn_model = ChessCNNModel(stochastic_depth=0.3)
cnn_path = base_dir / "model/cnn_stochastic_depth.pth"
cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
cnn_model.to(device)
cnn_model.eval()

# MLP: Original architecture retrained on D20 data
mlp_model = ChessModel(780)
mlp_path = base_dir / "model/mlp_retrained.pth"
mlp_model.load_state_dict(torch.load(mlp_path, map_location=device))
mlp_model.to(device)
mlp_model.eval()

print(f"Ensemble loaded on {device}: CNN (1.9M params) + MLP (12.3M params)")

def _minmax_normalize(val, vmin, vmax):
    """Normalize a single value given precomputed min/max."""
    return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5


# Precomputed min/max from validation set for ensemble normalization
CNN_MIN, CNN_MAX = 0.0219, 21.53
MLP_MIN, MLP_MAX = 0.599, 5.439

# Load percentile calibration (99 breakpoints → 100 buckets)
import bisect
calibration_path = base_dir / "model/complexity_calibration.json"
with open(calibration_path) as f:
    _calibration = json.load(f)
BREAKPOINTS = _calibration["breakpoints"]
print(f"Loaded {len(BREAKPOINTS)} calibration breakpoints")


def get_ensemble_prediction(fen: str) -> float:
    """Get raw ensemble prediction for a FEN. Returns value in [0, ~1] range."""
    cnn_tensor = fen_to_tensor(fen).unsqueeze(0).to(device)
    mlp_tensor = torch.tensor(fen_encoder(fen), dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        cnn_pred = cnn_model(cnn_tensor).squeeze().item()
        mlp_pred = mlp_model(mlp_tensor).squeeze().item() * 100  # sigmoid → raw scale

    cnn_norm = _minmax_normalize(cnn_pred, CNN_MIN, CNN_MAX)
    mlp_norm = _minmax_normalize(mlp_pred, MLP_MIN, MLP_MAX)
    return (cnn_norm + mlp_norm) / 2


def get_complexity_score(fen: str) -> int:
    """Get the complexity score (1-100) for a given FEN using the ensemble model.

    Uses percentile-based calibration: score N means the position is more complex
    than N% of positions in the calibration dataset (35,739 OTB games, Elo 2000+).

    1 = simplest positions (forced moves, clear advantages)
    100 = most complex positions (sharp middlegames, unclear compensation)
    """
    ensemble_pred = get_ensemble_prediction(fen)
    score = bisect.bisect_left(BREAKPOINTS, ensemble_pred) + 1
    return max(1, min(100, score))


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Elocator API!",
    }

@app.post("/complexity/")
def get_complexity(request: ComplexityRequest):
    '''Get the complexity score for a given FEN string.'''
    response = {
        "complexity_score": get_complexity_score(request.fen)
    }
    return response

@app.post("/analyze-game/")
def analyze_game(request: GameRequest):
    '''Analyze a game for complexity scores and other metrics.'''
    headers, FENs = parse_pgn(request.pgn)
    complexities = [get_complexity_score(fen) for fen in FENs]
    position_eval = analyze_positions(FENs)

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

    parser = argparse.ArgumentParser(description="Run the FastAPI application")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the FastAPI application on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to run the FastAPI application on")
    args = parser.parse_args()

    uvicorn.run("app:app", host=args.host, port=args.port)
