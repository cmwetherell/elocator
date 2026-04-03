"""
Elo Estimation Pipeline
=======================
End-to-end pipeline that estimates a player's Elo from a game PGN by comparing
their actual move accuracy (Stockfish) against the ensemble's expected complexity.

Stages:
  1. gather  — Parse PGN games, run Stockfish + ensemble, extract per-player features
  2. train   — Train a regression model (features → Elo) on gathered data
  3. predict — Given a new PGN, predict both players' Elos

Usage:
  python elo_pipeline.py gather --games 200 --depth 12
  python elo_pipeline.py train
  python elo_pipeline.py predict --pgn "1. e4 e5 ..."
"""

import sys
import os
import time
import json
import bisect
import math
from pathlib import Path
from collections import defaultdict

import chess
import chess.pgn
import chess.engine
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import fen_to_tensor, fen_encoder, get_win_percent
from model_cnn import ChessCNNModel, AttentionCNN

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "api" / "model"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PGN_PATH = DATA_DIR / "filtered.pgn"
OUTPUT_DIR = DATA_DIR / "elo_pipeline"
FEATURES_FILE = OUTPUT_DIR / "player_features.jsonl"
RAW_GAMES_FILE = OUTPUT_DIR / "raw_game_data.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "gather_checkpoint.json"
ELO_MODEL_FILE = OUTPUT_DIR / "elo_model.pth"
ELO_SCALER_FILE = OUTPUT_DIR / "elo_scaler.json"

STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

# ---------------------------------------------------------------------------
# MLP architecture (same as app.py / batch scorer)
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
# Load ensemble models + calibration
# ---------------------------------------------------------------------------
_models_cache = None

def load_ensemble():
    global _models_cache
    if _models_cache is not None:
        return _models_cache

    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            torch.tensor([], device="mps")
            device = "mps"
        except RuntimeError:
            pass
    print(f"Ensemble device: {device}")

    cnn_sd = ChessCNNModel(stochastic_depth=0.3)
    cnn_sd.load_state_dict(torch.load(MODEL_DIR / "cnn_stochastic_depth.pth", map_location=device))
    cnn_sd.to(device).eval()

    attn_cnn = AttentionCNN(stochastic_depth=0.3)
    attn_cnn.load_state_dict(torch.load(MODEL_DIR / "attention_cnn.pth", map_location=device))
    attn_cnn.to(device).eval()

    mlp = ChessModel(780)
    mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_retrained.pth", map_location=device))
    mlp.to(device).eval()

    with open(MODEL_DIR / "complexity_calibration.json") as f:
        cal = json.load(f)

    _models_cache = (cnn_sd, attn_cnn, mlp, cal, device)
    return _models_cache


def _minmax_norm(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5


@torch.no_grad()
def batch_ensemble_predict(fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=256):
    """Score FENs with ensemble. Returns list of dicts with raw + normalized predictions."""
    sd_min, sd_max = cal["sd_min"], cal["sd_max"]
    attn_min, attn_max = cal["attn_min"], cal["attn_max"]
    mlp_min, mlp_max = cal["mlp_min"], cal["mlp_max"]
    breakpoints = cal["breakpoints"]

    results = []
    for i in range(0, len(fens), batch_size):
        chunk = fens[i:i + batch_size]
        cnn_tensors = torch.stack([fen_to_tensor(f) for f in chunk]).to(device)
        mlp_vectors = torch.tensor([fen_encoder(f) for f in chunk], dtype=torch.float32).to(device)

        sd_preds = cnn_sd(cnn_tensors).squeeze(1).cpu().numpy()
        attn_preds = attn_cnn(cnn_tensors).squeeze(1).cpu().numpy()
        mlp_preds = mlp(mlp_vectors).squeeze(1).cpu().numpy() * 100

        for j in range(len(chunk)):
            sd_raw = float(sd_preds[j])
            attn_raw = float(attn_preds[j])
            mlp_raw = float(mlp_preds[j])
            sd_norm = _minmax_norm(sd_raw, sd_min, sd_max)
            attn_norm = _minmax_norm(attn_raw, attn_min, attn_max)
            mlp_norm = _minmax_norm(mlp_raw, mlp_min, mlp_max)
            ensemble = (sd_norm + attn_norm + mlp_norm) / 3
            complexity = max(1, min(100, bisect.bisect_left(breakpoints, ensemble) + 1))
            results.append({
                "ensemble": ensemble,
                "complexity": complexity,
                "sd_raw": sd_raw,
                "attn_raw": attn_raw,
                "mlp_raw": mlp_raw,
            })
    return results


# ---------------------------------------------------------------------------
# Stockfish evaluation
# ---------------------------------------------------------------------------
def stockfish_eval_game(game, engine, depth=12):
    """Evaluate every position in a game with Stockfish.

    Returns list of dicts, one per MOVE (not position):
      {fen_before, fen_after, move_san, color, cp_before, cp_after, actual_accuracy_loss}
    """
    board = game.board()
    moves_data = []

    # Evaluate starting position
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    prev_cp_white = info["score"].white().score(mate_score=100000)

    for move in game.mainline_moves():
        fen_before = board.fen()
        color = "white" if board.turn == chess.WHITE else "black"
        move_san = board.san(move)

        cp_before_white = prev_cp_white

        board.push(move)
        fen_after = board.fen()

        info = engine.analyse(board, chess.engine.Limit(depth=depth))
        cp_after_white = info["score"].white().score(mate_score=100000)

        # Convert to mover's perspective
        if color == "white":
            cp_before = cp_before_white
            cp_after = cp_after_white
        else:
            cp_before = -cp_before_white
            cp_after = -cp_after_white

        # Win% loss from mover's perspective (positive = bad move)
        win_pct_before = get_win_percent(cp_before)
        win_pct_after = get_win_percent(cp_after)
        actual_loss = max(0.0, win_pct_before - win_pct_after)

        moves_data.append({
            "fen_before": fen_before,
            "fen_after": fen_after,
            "move_san": move_san,
            "color": color,
            "cp_before_white": cp_before_white,
            "cp_after_white": cp_after_white,
            "actual_accuracy_loss": round(actual_loss, 4),
        })

        prev_cp_white = cp_after_white

    return moves_data


# ---------------------------------------------------------------------------
# Feature extraction: per-player features from one game
# ---------------------------------------------------------------------------
def _streak_max(vals, threshold):
    """Longest consecutive run where val < threshold."""
    best = cur = 0
    for v in vals:
        if v < threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _safe_slope(values):
    """Linear regression slope of values vs move index. Returns 0 if too few."""
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    # Simple least-squares slope
    xm = x.mean()
    ym = y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


from scipy.stats import skew, kurtosis as scipy_kurtosis


def extract_player_features(moves_data, ensemble_preds, headers, color):
    """Extract features for one player (white or black) from a single game.

    moves_data: list of per-move dicts from stockfish_eval_game
    ensemble_preds: list of ensemble predictions for each fen_before (aligned with moves_data)
    """
    player_moves = [(m, e) for m, e in zip(moves_data, ensemble_preds) if m["color"] == color]

    if len(player_moves) < 5:
        return None  # too few moves

    actual_losses = [m["actual_accuracy_loss"] for m, _ in player_moves]
    expected_complexities = [e["ensemble"] for _, e in player_moves]
    complexity_scores = [e["complexity"] for _, e in player_moves]
    cp_before_values = [m["cp_before_white"] for m, _ in player_moves]

    n_moves = len(player_moves)
    losses_arr = np.array(actual_losses)
    complexity_arr = np.array(complexity_scores)
    ensemble_arr = np.array(expected_complexities)

    # ===================================================================
    # GROUP 1: Core accuracy stats (original)
    # ===================================================================
    mean_actual = float(losses_arr.mean())
    std_actual = float(losses_arr.std())
    median_loss = float(np.median(losses_arr))
    p75_loss = float(np.percentile(losses_arr, 75))
    p90_loss = float(np.percentile(losses_arr, 90))
    max_loss = float(losses_arr.max())

    # ===================================================================
    # GROUP 2: Distribution shape
    # ===================================================================
    loss_skewness = float(skew(losses_arr)) if n_moves >= 3 else 0.0
    loss_kurtosis = float(scipy_kurtosis(losses_arr)) if n_moves >= 4 else 0.0
    coeff_variation = std_actual / mean_actual if mean_actual > 0.01 else 0.0
    # Interquartile range
    iqr_loss = float(np.percentile(losses_arr, 75) - np.percentile(losses_arr, 25))

    # ===================================================================
    # GROUP 3: Expected complexity features
    # ===================================================================
    mean_expected = float(ensemble_arr.mean())
    mean_complexity_score = float(complexity_arr.mean())
    std_complexity = float(complexity_arr.std())
    max_complexity = float(complexity_arr.max())

    # ===================================================================
    # GROUP 4: Relative performance (actual vs expected)
    # ===================================================================
    ratios = []
    residuals = []
    for (m, e) in player_moves:
        exp = e["ensemble"]
        act = m["actual_accuracy_loss"]
        if exp > 0.01:
            ratios.append(act / (exp * 10))
        residuals.append(act - exp * 10)

    mean_ratio = float(np.mean(ratios)) if ratios else 0
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))

    # Complexity-weighted accuracy: weight each loss by position complexity
    weights = complexity_arr / complexity_arr.sum() if complexity_arr.sum() > 0 else np.ones(n_moves) / n_moves
    complexity_weighted_loss = float((losses_arr * weights).sum())

    # ===================================================================
    # GROUP 5: Move quality buckets
    # ===================================================================
    perfect_moves = int((losses_arr < 0.1).sum())
    good_moves = int((losses_arr < 1.0).sum())
    inaccuracies = int(((losses_arr >= 1.0) & (losses_arr < 3.0)).sum())
    mistakes = int(((losses_arr >= 3.0) & (losses_arr < 7.0)).sum())
    blunders = int((losses_arr >= 7.0).sum())
    # Finer-grained: very small inaccuracies (0.1-0.5) — subtle errors only top players avoid
    minor_inaccuracies = int(((losses_arr >= 0.1) & (losses_arr < 0.5)).sum())
    # Major blunders (>15% win equity)
    major_blunders = int((losses_arr >= 15.0).sum())

    # ===================================================================
    # GROUP 6: Complexity-stratified accuracy
    # ===================================================================
    # Very complex (>=75), complex (50-74), moderate (25-49), simple (<25)
    very_complex = [(m, e) for m, e in player_moves if e["complexity"] >= 75]
    complex_mid = [(m, e) for m, e in player_moves if 50 <= e["complexity"] < 75]
    moderate = [(m, e) for m, e in player_moves if 25 <= e["complexity"] < 50]
    simple = [(m, e) for m, e in player_moves if e["complexity"] < 25]

    mean_loss_very_complex = float(np.mean([m["actual_accuracy_loss"] for m, _ in very_complex])) if very_complex else 0
    mean_loss_complex_mid = float(np.mean([m["actual_accuracy_loss"] for m, _ in complex_mid])) if complex_mid else 0
    mean_loss_moderate = float(np.mean([m["actual_accuracy_loss"] for m, _ in moderate])) if moderate else 0
    mean_loss_simple = float(np.mean([m["actual_accuracy_loss"] for m, _ in simple])) if simple else 0

    n_very_complex = len(very_complex)
    n_complex_mid = len(complex_mid)
    n_simple = len(simple)

    # Perfect move rate in complex vs simple positions
    pct_perfect_complex = (sum(1 for m, _ in very_complex + complex_mid if m["actual_accuracy_loss"] < 0.1)
                           / max(1, len(very_complex) + len(complex_mid)))
    pct_perfect_simple = (sum(1 for m, _ in simple + moderate if m["actual_accuracy_loss"] < 0.1)
                          / max(1, len(simple) + len(moderate)))

    # Complex-to-simple loss ratio (higher = struggles more in complex positions)
    loss_ratio_complex_simple = (mean_loss_very_complex / mean_loss_simple) if mean_loss_simple > 0.01 else 0

    # ===================================================================
    # GROUP 7: Game phase accuracy (opening / middlegame / endgame)
    # ===================================================================
    # Split by move number: opening (1-15), middlegame (16-30), late (31+)
    opening_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if i < 8]  # ~first 15 half-moves
    middle_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if 8 <= i < 20]
    late_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if i >= 20]

    mean_loss_opening = float(np.mean(opening_losses)) if opening_losses else 0
    mean_loss_middlegame = float(np.mean(middle_losses)) if middle_losses else 0
    mean_loss_endgame = float(np.mean(late_losses)) if late_losses else 0

    pct_perfect_opening = (sum(1 for l in opening_losses if l < 0.1) / max(1, len(opening_losses)))
    pct_perfect_endgame = (sum(1 for l in late_losses if l < 0.1) / max(1, len(late_losses)))

    # ===================================================================
    # GROUP 8: Accuracy trend (slope over time)
    # ===================================================================
    # Positive slope = getting worse as game goes on (fatigue)
    accuracy_slope = _safe_slope(actual_losses)
    # Complexity trend — are they steering into complex or simple positions?
    complexity_slope = _safe_slope(complexity_scores)

    # ===================================================================
    # GROUP 9: Pressure performance
    # ===================================================================
    # Performance when behind (eval worse than -100cp from their perspective)
    behind_moves = []
    ahead_moves = []
    equal_moves = []
    for m, e in player_moves:
        cp = m["cp_before_white"]
        # Convert to player perspective
        player_cp = cp if color == "white" else -cp
        if player_cp < -100:
            behind_moves.append(m["actual_accuracy_loss"])
        elif player_cp > 100:
            ahead_moves.append(m["actual_accuracy_loss"])
        else:
            equal_moves.append(m["actual_accuracy_loss"])

    mean_loss_behind = float(np.mean(behind_moves)) if behind_moves else 0
    mean_loss_ahead = float(np.mean(ahead_moves)) if ahead_moves else 0
    mean_loss_equal = float(np.mean(equal_moves)) if equal_moves else 0
    pct_moves_behind = len(behind_moves) / n_moves
    pct_moves_ahead = len(ahead_moves) / n_moves

    # ===================================================================
    # GROUP 10: Streak / consistency features
    # ===================================================================
    longest_good_streak = _streak_max(actual_losses, 1.0)
    longest_perfect_streak = _streak_max(actual_losses, 0.1)
    # Normalized by game length
    good_streak_ratio = longest_good_streak / n_moves
    perfect_streak_ratio = longest_perfect_streak / n_moves

    # Count of "back-to-back" errors (consecutive moves with loss > 2.0)
    consecutive_errors = sum(1 for i in range(1, n_moves) if actual_losses[i] > 2.0 and actual_losses[i-1] > 2.0)

    # ===================================================================
    # GROUP 11: Interaction features
    # ===================================================================
    # Perfect rate * mean complexity — being perfect in hard positions
    perfect_times_complexity = (perfect_moves / n_moves) * mean_complexity_score
    # Blunder rate * inverse complexity — blundering in easy positions is worse
    blunder_in_simple = sum(1 for m, e in player_moves
                           if m["actual_accuracy_loss"] >= 7.0 and e["complexity"] < 50)
    blunder_in_simple_rate = blunder_in_simple / max(1, len(simple) + len(moderate))

    # ===================================================================
    # GROUP 12: Eval trajectory features
    # ===================================================================
    # How much does the eval swing during the player's moves?
    eval_changes = []
    for m, _ in player_moves:
        cp_before = m["cp_before_white"] if color == "white" else -m["cp_before_white"]
        cp_after = m["cp_after_white"] if color == "white" else -m["cp_after_white"]
        eval_changes.append(cp_after - cp_before)
    mean_eval_change = float(np.mean(eval_changes))
    std_eval_change = float(np.std(eval_changes))

    # ===================================================================
    # Elo (target)
    # ===================================================================
    elo = headers.get(f"{'White' if color == 'white' else 'Black'}Elo")
    try:
        elo = int(elo)
    except (TypeError, ValueError):
        return None

    return {
        "elo": elo,
        "color": color,
        "player": headers.get("White" if color == "white" else "Black", "?"),
        "opponent_elo": _safe_int(headers.get(f"{'Black' if color == 'white' else 'White'}Elo")),
        "result": headers.get("Result", "?"),

        # G1: Core accuracy
        "n_moves": n_moves,
        "game_length": len(moves_data),
        "mean_actual_loss": round(mean_actual, 4),
        "std_actual_loss": round(std_actual, 4),
        "median_loss": round(median_loss, 4),
        "p75_loss": round(p75_loss, 4),
        "p90_loss": round(p90_loss, 4),
        "max_loss": round(max_loss, 4),

        # G2: Distribution shape
        "loss_skewness": round(loss_skewness, 4),
        "loss_kurtosis": round(loss_kurtosis, 4),
        "coeff_variation": round(coeff_variation, 4),
        "iqr_loss": round(iqr_loss, 4),

        # G3: Expected complexity
        "mean_expected_ensemble": round(mean_expected, 4),
        "mean_complexity_score": round(mean_complexity_score, 2),
        "std_complexity": round(std_complexity, 2),
        "max_complexity": round(max_complexity, 2),

        # G4: Relative performance
        "mean_ratio": round(mean_ratio, 4),
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
        "complexity_weighted_loss": round(complexity_weighted_loss, 4),

        # G5: Move quality buckets
        "pct_perfect": round(perfect_moves / n_moves, 4),
        "pct_good": round(good_moves / n_moves, 4),
        "pct_inaccuracy": round(inaccuracies / n_moves, 4),
        "pct_mistake": round(mistakes / n_moves, 4),
        "pct_blunder": round(blunders / n_moves, 4),
        "pct_minor_inaccuracy": round(minor_inaccuracies / n_moves, 4),
        "pct_major_blunder": round(major_blunders / n_moves, 4),

        # G6: Complexity-stratified
        "mean_loss_very_complex": round(mean_loss_very_complex, 4),
        "mean_loss_complex_mid": round(mean_loss_complex_mid, 4),
        "mean_loss_moderate": round(mean_loss_moderate, 4),
        "mean_loss_simple": round(mean_loss_simple, 4),
        "n_very_complex": n_very_complex,
        "n_complex_mid": n_complex_mid,
        "n_simple": n_simple,
        "pct_perfect_complex": round(pct_perfect_complex, 4),
        "pct_perfect_simple": round(pct_perfect_simple, 4),
        "loss_ratio_complex_simple": round(loss_ratio_complex_simple, 4),

        # G7: Game phase
        "mean_loss_opening": round(mean_loss_opening, 4),
        "mean_loss_middlegame": round(mean_loss_middlegame, 4),
        "mean_loss_endgame": round(mean_loss_endgame, 4),
        "pct_perfect_opening": round(pct_perfect_opening, 4),
        "pct_perfect_endgame": round(pct_perfect_endgame, 4),

        # G8: Trends
        "accuracy_slope": round(accuracy_slope, 6),
        "complexity_slope": round(complexity_slope, 4),

        # G9: Pressure
        "mean_loss_behind": round(mean_loss_behind, 4),
        "mean_loss_ahead": round(mean_loss_ahead, 4),
        "mean_loss_equal": round(mean_loss_equal, 4),
        "pct_moves_behind": round(pct_moves_behind, 4),
        "pct_moves_ahead": round(pct_moves_ahead, 4),

        # G10: Streaks
        "longest_good_streak": longest_good_streak,
        "longest_perfect_streak": longest_perfect_streak,
        "good_streak_ratio": round(good_streak_ratio, 4),
        "perfect_streak_ratio": round(perfect_streak_ratio, 4),
        "consecutive_errors": consecutive_errors,

        # G11: Interactions
        "perfect_times_complexity": round(perfect_times_complexity, 4),
        "blunder_in_simple_rate": round(blunder_in_simple_rate, 4),

        # G12: Eval trajectory
        "mean_eval_change": round(mean_eval_change, 2),
        "std_eval_change": round(std_eval_change, 2),
    }


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Stage 1: Gather — process games and extract features
# ---------------------------------------------------------------------------
GAME_INDEX_PATH = DATA_DIR / "game_index.json"
SAMPLED_INDICES_FILE = OUTPUT_DIR / "sampled_game_indices.json"


def _load_game_index():
    """Load the byte-offset index for random access into filtered.pgn."""
    with open(GAME_INDEX_PATH) as f:
        return json.load(f)


def _read_game_at_offset(pgn_path, offset):
    """Read a single game from a PGN file at a given byte offset."""
    with open(pgn_path) as f:
        f.seek(offset)
        return chess.pgn.read_game(f)


def _load_sampled_indices():
    """Load set of already-sampled game indices to avoid duplicates."""
    if SAMPLED_INDICES_FILE.exists():
        with open(SAMPLED_INDICES_FILE) as f:
            return set(json.load(f))
    return set()


def _save_sampled_indices(indices):
    with open(SAMPLED_INDICES_FILE, "w") as f:
        json.dump(sorted(indices), f)


def _process_single_game(game, engine, cnn_sd, attn_cnn, mlp, cal, device, depth):
    """Process a single game: Stockfish eval + ensemble + feature extraction.
    Returns (feature_dicts, raw_game_record) where raw_game_record contains
    per-move Stockfish + ensemble data for future re-extraction."""
    headers = dict(game.headers)

    # Skip games without valid Elo
    w_elo = headers.get("WhiteElo")
    b_elo = headers.get("BlackElo")
    if not w_elo or not b_elo:
        return [], None
    try:
        int(w_elo)
        int(b_elo)
    except ValueError:
        return [], None

    # Stockfish analysis
    try:
        moves_data = stockfish_eval_game(game, engine, depth=depth)
    except Exception as e:
        print(f"  Stockfish error: {e}")
        return [], None

    if len(moves_data) < 10:
        return [], None

    # Ensemble predictions
    before_fens = [m["fen_before"] for m in moves_data]
    ensemble_preds = batch_ensemble_predict(
        before_fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=256
    )

    # Build per-move raw records (merge Stockfish + ensemble)
    raw_moves = []
    for m, e in zip(moves_data, ensemble_preds):
        raw_moves.append({
            "fen_before": m["fen_before"],
            "fen_after": m["fen_after"],
            "move_san": m["move_san"],
            "color": m["color"],
            "cp_before_white": m["cp_before_white"],
            "cp_after_white": m["cp_after_white"],
            "actual_accuracy_loss": m["actual_accuracy_loss"],
            "ensemble": round(e["ensemble"], 4),
            "complexity": e["complexity"],
            "sd_raw": round(e["sd_raw"], 4),
            "attn_raw": round(e["attn_raw"], 4),
            "mlp_raw": round(e["mlp_raw"], 4),
        })

    raw_game_record = {
        "headers": headers,
        "depth": depth,
        "moves": raw_moves,
    }

    # Extract summary features for each player
    results = []
    for color in ["white", "black"]:
        features = extract_player_features(moves_data, ensemble_preds, headers, color)
        if features is not None:
            results.append(features)
    return results, raw_game_record


def gather(n_games=200, depth=12, time_limit_minutes=None, random_sample=False):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load ensemble
    cnn_sd, attn_cnn, mlp, cal, device = load_ensemble()

    # Open Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, timeout=30)
    engine.configure({"Threads": 4})

    out_f = open(FEATURES_FILE, "a")
    raw_f = open(RAW_GAMES_FILE, "a")
    t0 = time.time()
    deadline = t0 + time_limit_minutes * 60 if time_limit_minutes else None

    if random_sample:
        # Random sampling mode: pick random games from across the full PGN
        print("Loading game index for random sampling...")
        game_index = _load_game_index()
        total_games_in_pgn = len(game_index)
        already_sampled = _load_sampled_indices()
        print(f"  {total_games_in_pgn:,} games in PGN, {len(already_sampled):,} already sampled")

        # Build candidate pool (exclude already sampled)
        candidates = [i for i in range(total_games_in_pgn) if i not in already_sampled]
        rng = np.random.RandomState(int(time.time()) % 2**31)
        rng.shuffle(candidates)

        games_done = 0
        features_written = 0
        candidate_idx = 0

        print(f"Gathering data: up to {n_games} games, Stockfish depth={depth}, random sampling")

        while games_done < n_games and candidate_idx < len(candidates):
            if deadline and time.time() > deadline:
                print("Time limit reached.")
                break

            gi = candidates[candidate_idx]
            candidate_idx += 1
            offset = game_index[gi]

            game = _read_game_at_offset(PGN_PATH, offset)
            if game is None:
                continue

            results, raw_record = _process_single_game(game, engine, cnn_sd, attn_cnn, mlp, cal, device, depth)
            for feat in results:
                out_f.write(json.dumps(feat) + "\n")
                features_written += 1
            if raw_record is not None:
                raw_f.write(json.dumps(raw_record) + "\n")

            if results:
                already_sampled.add(gi)
                games_done += 1

            if games_done % 10 == 0 and games_done > 0:
                elapsed = time.time() - t0
                rate = games_done / elapsed * 60
                out_f.flush()
                raw_f.flush()
                _save_sampled_indices(already_sampled)
                # Show Elo range of recent samples
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump({
                        "games_completed": games_done,
                        "total_sampled": len(already_sampled),
                        "features_written": features_written,
                        "mode": "random",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, f, indent=2)
                remaining = ""
                if deadline:
                    rem_min = (deadline - time.time()) / 60
                    remaining = f" | {rem_min:.0f}m remaining"
                print(f"  [{elapsed/60:.1f}m] {games_done}/{n_games} games | "
                      f"{features_written} player records | {rate:.1f} games/min{remaining}")

        _save_sampled_indices(already_sampled)

    else:
        # Sequential mode (original behavior)
        start_game = 0
        byte_offset = 0
        if CHECKPOINT_FILE.exists():
            with open(CHECKPOINT_FILE) as f:
                ckpt = json.load(f)
            start_game = ckpt.get("games_completed", 0)
            byte_offset = ckpt.get("byte_offset", 0)
            if ckpt.get("mode") != "random":
                print(f"Resuming sequential from game {start_game}")

        games_done = 0
        games_total = start_game
        features_written = 0

        print(f"Gathering data: {n_games} games, Stockfish depth={depth}, sequential")

        with open(PGN_PATH) as pgn_f:
            if byte_offset > 0:
                pgn_f.seek(byte_offset)

            while games_done < n_games:
                if deadline and time.time() > deadline:
                    print("Time limit reached.")
                    break

                game = chess.pgn.read_game(pgn_f)
                if game is None:
                    print("End of PGN file.")
                    break

                current_offset = pgn_f.tell()

                results, raw_record = _process_single_game(game, engine, cnn_sd, attn_cnn, mlp, cal, device, depth)
                for feat in results:
                    out_f.write(json.dumps(feat) + "\n")
                    features_written += 1
                if raw_record is not None:
                    raw_f.write(json.dumps(raw_record) + "\n")

                if results:
                    games_done += 1
                games_total += 1

                if games_done % 10 == 0 and games_done > 0:
                    elapsed = time.time() - t0
                    rate = games_done / elapsed * 60
                    out_f.flush()
                    with open(CHECKPOINT_FILE, "w") as f:
                        json.dump({
                            "games_completed": games_total,
                            "byte_offset": current_offset,
                            "features_written": features_written,
                            "mode": "sequential",
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }, f, indent=2)
                    print(f"  [{elapsed/60:.1f}m] {games_done}/{n_games} games | "
                          f"{features_written} player records | {rate:.1f} games/min")

        # Final checkpoint for sequential
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "games_completed": games_total,
                "byte_offset": current_offset if games_done > 0 else byte_offset,
                "features_written": features_written,
                "mode": "sequential",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    engine.quit()
    out_f.close()
    raw_f.close()
    elapsed = time.time() - t0

    # Final checkpoint for random
    if random_sample:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump({
                "games_completed": games_done,
                "total_sampled": len(already_sampled),
                "features_written": features_written,
                "mode": "random",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }, f, indent=2)

    print(f"\nGather complete: {games_done} games, {features_written} player records in {elapsed/60:.1f}m")
    print(f"Output: {FEATURES_FILE}")
    print(f"Raw data: {RAW_GAMES_FILE}")


# ---------------------------------------------------------------------------
# Stage 1b: Re-extract — recompute features from saved raw game data
# ---------------------------------------------------------------------------
def reextract():
    """Re-extract player features from raw_game_data.jsonl without re-running Stockfish."""
    print("Re-extracting features from raw game data...")
    raw_count = 0
    features_written = 0

    # Overwrite features file
    with open(RAW_GAMES_FILE) as raw_f, open(FEATURES_FILE, "w") as out_f:
        for line in raw_f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            headers = record["headers"]
            moves = record["moves"]

            # Reconstruct moves_data and ensemble_preds in the format
            # extract_player_features expects
            moves_data = []
            ensemble_preds = []
            for m in moves:
                moves_data.append({
                    "fen_before": m["fen_before"],
                    "fen_after": m["fen_after"],
                    "move_san": m["move_san"],
                    "color": m["color"],
                    "cp_before_white": m["cp_before_white"],
                    "cp_after_white": m["cp_after_white"],
                    "actual_accuracy_loss": m["actual_accuracy_loss"],
                })
                ensemble_preds.append({
                    "ensemble": m["ensemble"],
                    "complexity": m["complexity"],
                    "sd_raw": m["sd_raw"],
                    "attn_raw": m["attn_raw"],
                    "mlp_raw": m["mlp_raw"],
                })

            for color in ["white", "black"]:
                features = extract_player_features(moves_data, ensemble_preds, headers, color)
                if features is not None:
                    out_f.write(json.dumps(features) + "\n")
                    features_written += 1

            raw_count += 1
            if raw_count % 100 == 0:
                print(f"  {raw_count} games re-extracted, {features_written} player records")

    print(f"\nRe-extraction complete: {raw_count} games → {features_written} player records")
    print(f"Output: {FEATURES_FILE}")


# ---------------------------------------------------------------------------
# Stage 2: Train — fit a model on gathered features
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    # G1: Core accuracy
    "n_moves", "game_length",
    "mean_actual_loss", "std_actual_loss", "median_loss", "p75_loss", "p90_loss", "max_loss",
    # G2: Distribution shape
    "loss_skewness", "loss_kurtosis", "coeff_variation", "iqr_loss",
    # G3: Expected complexity
    "mean_expected_ensemble", "mean_complexity_score", "std_complexity", "max_complexity",
    # G4: Relative performance
    "mean_ratio", "mean_residual", "std_residual", "complexity_weighted_loss",
    # G5: Move quality buckets
    "pct_perfect", "pct_good", "pct_inaccuracy", "pct_mistake", "pct_blunder",
    "pct_minor_inaccuracy", "pct_major_blunder",
    # G6: Complexity-stratified
    "mean_loss_very_complex", "mean_loss_complex_mid", "mean_loss_moderate", "mean_loss_simple",
    "n_very_complex", "n_complex_mid", "n_simple",
    "pct_perfect_complex", "pct_perfect_simple", "loss_ratio_complex_simple",
    # G7: Game phase
    "mean_loss_opening", "mean_loss_middlegame", "mean_loss_endgame",
    "pct_perfect_opening", "pct_perfect_endgame",
    # G8: Trends
    "accuracy_slope", "complexity_slope",
    # G9: Pressure
    "mean_loss_behind", "mean_loss_ahead", "mean_loss_equal",
    "pct_moves_behind", "pct_moves_ahead",
    # G10: Streaks
    "longest_good_streak", "longest_perfect_streak",
    "good_streak_ratio", "perfect_streak_ratio", "consecutive_errors",
    # G11: Interactions
    "perfect_times_complexity", "blunder_in_simple_rate",
    # G12: Eval trajectory
    "mean_eval_change", "std_eval_change",
    # Barometer: random noise (should rank last if features are real)
    "random_noise",
]


class EloPredictor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_features():
    """Load gathered feature records."""
    records = []
    with open(FEATURES_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def records_to_arrays(records):
    """Convert records to feature matrix X and target vector y."""
    rng = np.random.RandomState(123)
    X = []
    y = []
    for r in records:
        row = []
        for col in FEATURE_COLS:
            if col == "random_noise":
                row.append(rng.randn())
            else:
                row.append(r.get(col, 0) or 0)
        X.append(row)
        y.append(r["elo"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_model(method="lgbm"):
    print("Loading features...")
    records = load_features()
    print(f"Loaded {len(records)} player-game records")

    X, y = records_to_arrays(records)
    print(f"Feature matrix: {X.shape}, Target range: [{y.min():.0f}, {y.max():.0f}]")

    # 3-way split: 70% train, 15% early-stopping holdout, 15% final test
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    train_idx = idx[:train_end]
    val_idx = idx[train_end:val_end]
    test_idx = idx[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print(f"Train: {len(X_train)}, Val (early stop): {len(X_val)}, Test (holdout): {len(X_test)}")

    if method == "lgbm":
        _train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test)
    else:
        _train_mlp(X_train, y_train, X_val, y_val, X_test, y_test)


def _train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test):
    import lightgbm as lgb
    import pickle

    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_ds)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 20,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "verbose": -1,
    }

    print("\nTraining LightGBM...")
    callbacks = [
        lgb.log_evaluation(period=100),
        lgb.early_stopping(stopping_rounds=100),
    ]
    model = lgb.train(
        params,
        train_ds,
        num_boost_round=5000,
        valid_sets=[train_ds, val_ds],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Save model
    lgbm_model_path = OUTPUT_DIR / "elo_model_lgbm.txt"
    model.save_model(str(lgbm_model_path))

    # Also save as pickle for predict_from_pgn
    lgbm_pkl_path = OUTPUT_DIR / "elo_model_lgbm.pkl"
    with open(lgbm_pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Save scaler info (LightGBM doesn't need normalization, but predict needs feature cols)
    scaler = {
        "method": "lgbm",
        "feature_cols": FEATURE_COLS,
    }
    with open(ELO_SCALER_FILE, "w") as f:
        json.dump(scaler, f, indent=2)

    # Evaluate on holdout test set
    test_pred = model.predict(X_test)
    val_pred = model.predict(X_val)
    train_pred = model.predict(X_train)

    for name, pred, true in [("Train", train_pred, y_train),
                              ("Val", val_pred, y_val),
                              ("Test", test_pred, y_test)]:
        mae = np.abs(pred - true).mean()
        rmse = np.sqrt(((pred - true) ** 2).mean())
        corr = np.corrcoef(pred, true)[0, 1]
        print(f"  {name:5s}: MAE={mae:.0f}  RMSE={rmse:.0f}  Pearson={corr:.4f}")

    print(f"\n  Best iteration: {model.best_iteration}")
    print(f"  Model saved: {lgbm_model_path}")

    # Feature importance (gain)
    importance = model.feature_importance(importance_type="gain")
    ranked = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    print("\n  Feature importance (gain):")
    max_imp = max(importance) if max(importance) > 0 else 1
    # Find random_noise importance as barometer
    noise_imp = dict(zip(FEATURE_COLS, importance)).get("random_noise", 0)
    for name, imp in ranked:
        marker = " <<< RANDOM BAROMETER" if name == "random_noise" else ""
        above = " *" if imp > noise_imp and name != "random_noise" else ""
        bar = "#" * int(imp / max_imp * 30)
        print(f"    {name:32s} {imp:10.0f} {bar}{above}{marker}")
    n_above = sum(1 for _, imp in ranked if imp > noise_imp) - 1  # exclude noise itself
    print(f"\n  Features above random barometer: {n_above}/{len(FEATURE_COLS)-1}")

    # SHAP analysis
    print("\n  Computing SHAP values on test set...")
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # SHAP summary bar plot
    fig, ax = plt.subplots(figsize=(10, 14))
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS,
                      plot_type="bar", show=False, max_display=len(FEATURE_COLS))
    plt.title("SHAP Feature Importance — Elo Predictor (LightGBM)", fontsize=14)
    plt.tight_layout()
    shap_bar_path = OUTPUT_DIR / "shap_importance_bar.png"
    plt.savefig(shap_bar_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP bar plot: {shap_bar_path}")

    # SHAP beeswarm plot (shows direction of effect)
    fig, ax = plt.subplots(figsize=(10, 14))
    shap.summary_plot(shap_values, X_test, feature_names=FEATURE_COLS,
                      show=False, max_display=len(FEATURE_COLS))
    plt.title("SHAP Beeswarm — Elo Predictor (LightGBM)", fontsize=14)
    plt.tight_layout()
    shap_bee_path = OUTPUT_DIR / "shap_beeswarm.png"
    plt.savefig(shap_bee_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP beeswarm: {shap_bee_path}")


def _train_mlp(X_train, y_train, X_val, y_val, X_test, y_test):
    # Standardize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_std[X_std == 0] = 1.0
    y_mean = y_train.mean()
    y_std = y_train.std()

    X_train_n = (X_train - X_mean) / X_std
    X_val_n = (X_val - X_mean) / X_std
    X_test_n = (X_test - X_mean) / X_std
    y_train_n = (y_train - y_mean) / y_std
    y_val_n = (y_val - y_mean) / y_std

    scaler = {
        "method": "mlp",
        "X_mean": X_mean.tolist(),
        "X_std": X_std.tolist(),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "feature_cols": FEATURE_COLS,
    }
    with open(ELO_SCALER_FILE, "w") as f:
        json.dump(scaler, f, indent=2)

    Xt = torch.tensor(X_train_n)
    yt = torch.tensor(y_train_n).unsqueeze(1)
    Xv = torch.tensor(X_val_n)
    yv = torch.tensor(y_val_n).unsqueeze(1)

    model = EloPredictor(len(FEATURE_COLS))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(500):
        model.train()
        pred = model(Xt)
        loss = F.mse_loss(pred, yt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(Xv)
            val_loss = F.mse_loss(val_pred, yv)
            scheduler.step(val_loss)
            val_mae = (val_pred.squeeze() * y_std + y_mean - (yv.squeeze() * y_std + y_mean)).abs().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ELO_MODEL_FILE)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch:3d}: train_loss={loss.item():.4f} val_loss={val_loss.item():.4f} val_MAE={val_mae:.0f} Elo")

        if patience_counter >= 50:
            print(f"  Early stopping at epoch {epoch}")
            break

    # Final eval on test
    model.load_state_dict(torch.load(ELO_MODEL_FILE))
    model.eval()
    with torch.no_grad():
        test_pred = model(torch.tensor(X_test_n)).squeeze() * y_std + y_mean
        test_true = torch.tensor(y_test)
        mae = (test_pred - test_true).abs().mean().item()
        rmse = ((test_pred - test_true) ** 2).mean().sqrt().item()
        corr = np.corrcoef(test_pred.numpy(), test_true.numpy())[0, 1]

    print(f"\n=== FINAL RESULTS (holdout test) ===")
    print(f"  Test MAE:     {mae:.0f} Elo")
    print(f"  Test RMSE:    {rmse:.0f} Elo")
    print(f"  Test Pearson: {corr:.4f}")
    print(f"  Model saved: {ELO_MODEL_FILE}")


# ---------------------------------------------------------------------------
# Stage 3: Predict — estimate Elo from a PGN string
# ---------------------------------------------------------------------------
def predict_from_pgn(pgn_text, depth=12):
    """Given a PGN string, predict Elo for both players."""
    import io
    import pickle

    # Load scaler config
    with open(ELO_SCALER_FILE) as f:
        scaler = json.load(f)

    method = scaler.get("method", "mlp")

    if method == "lgbm":
        lgbm_pkl_path = OUTPUT_DIR / "elo_model_lgbm.pkl"
        with open(lgbm_pkl_path, "rb") as f:
            predict_model = pickle.load(f)
    else:
        predict_model = EloPredictor(len(FEATURE_COLS))
        predict_model.load_state_dict(torch.load(ELO_MODEL_FILE, map_location="cpu"))
        predict_model.eval()

    # Load ensemble
    cnn_sd, attn_cnn, mlp, cal, device = load_ensemble()

    # Parse PGN
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        print("Could not parse PGN")
        return
    headers = dict(game.headers)

    # Stockfish analysis
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, timeout=30)
    engine.configure({"Threads": 4})
    moves_data = stockfish_eval_game(game, engine, depth=depth)
    engine.quit()

    # Ensemble predictions
    before_fens = [m["fen_before"] for m in moves_data]
    ensemble_preds = batch_ensemble_predict(
        before_fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=256
    )

    # Extract & predict for each player
    results = {}
    for color in ["white", "black"]:
        features = extract_player_features(moves_data, ensemble_preds, headers, color)
        if features is None:
            print(f"  {color}: not enough data")
            continue

        row = [features.get(col, 0) or 0 for col in FEATURE_COLS]
        X = np.array([row], dtype=np.float32)

        if method == "lgbm":
            predicted_elo = float(predict_model.predict(X)[0])
        else:
            X_norm = (X - np.array(scaler["X_mean"], dtype=np.float32)) / np.array(scaler["X_std"], dtype=np.float32)
            with torch.no_grad():
                pred = predict_model(torch.tensor(X_norm, dtype=torch.float32)).item()
            predicted_elo = pred * scaler["y_std"] + scaler["y_mean"]

        player_name = headers.get("White" if color == "white" else "Black", "?")
        actual_elo = features.get("elo")

        results[color] = {
            "player": player_name,
            "predicted_elo": round(predicted_elo),
            "actual_elo": actual_elo,
            "mean_accuracy_loss": features["mean_actual_loss"],
            "mean_complexity": features["mean_complexity_score"],
            "pct_perfect": features["pct_perfect"],
            "pct_blunder": features["pct_blunder"],
        }

        print(f"\n  {color.upper()}: {player_name}")
        print(f"    Predicted Elo: {predicted_elo:.0f}")
        if actual_elo:
            print(f"    Actual Elo:    {actual_elo}")
            print(f"    Error:         {abs(predicted_elo - actual_elo):.0f}")
        print(f"    Mean accuracy loss: {features['mean_actual_loss']:.2f}%")
        print(f"    Mean complexity:    {features['mean_complexity_score']:.1f}")
        print(f"    Perfect moves:      {features['pct_perfect']*100:.1f}%")
        print(f"    Blunders:           {features['pct_blunder']*100:.1f}%")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Elo estimation pipeline")
    sub = parser.add_subparsers(dest="stage")

    g = sub.add_parser("gather", help="Gather features from PGN games")
    g.add_argument("--games", type=int, default=200, help="Number of games to process")
    g.add_argument("--depth", type=int, default=12, help="Stockfish depth")
    g.add_argument("--time", type=float, default=None, help="Time limit in minutes")
    g.add_argument("--random", action="store_true", help="Randomly sample games across the full PGN")

    sub.add_parser("reextract", help="Re-extract features from raw game data (no Stockfish needed)")

    t = sub.add_parser("train", help="Train Elo prediction model")
    t.add_argument("--method", type=str, default="lgbm", choices=["lgbm", "mlp"], help="Model type")

    p = sub.add_parser("predict", help="Predict Elo from PGN")
    p.add_argument("--pgn", type=str, required=True, help="PGN string or file path")
    p.add_argument("--depth", type=int, default=12, help="Stockfish depth")

    args = parser.parse_args()

    if args.stage == "gather":
        gather(n_games=args.games, depth=args.depth, time_limit_minutes=args.time, random_sample=args.random)
    elif args.stage == "reextract":
        reextract()
    elif args.stage == "train":
        train_model(method=args.method)
    elif args.stage == "predict":
        pgn_text = args.pgn
        if os.path.isfile(pgn_text):
            with open(pgn_text) as f:
                pgn_text = f.read()
        predict_from_pgn(pgn_text, depth=args.depth)
    else:
        parser.print_help()
