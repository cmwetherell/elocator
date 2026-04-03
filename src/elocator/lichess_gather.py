"""
Lichess Elo Pipeline — Gather features from Lichess PGN ZST dump.

Streams through the compressed PGN, filters for games with [%eval] annotations,
computes actual win% loss from the embedded evals, runs the ensemble for
expected complexity, and extracts per-player features.

No Stockfish needed — evals come straight from the PGN.

Usage:
  python lichess_gather.py --time 5   # run for 5 minutes
"""

import sys
import os
import io
import re
import time
import json
import bisect
import math
from pathlib import Path

import zstandard as zstd
import chess
import chess.pgn
import torch
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
LICHESS_DIR = DATA_DIR / "lichess"
OUTPUT_DIR = DATA_DIR / "elo_pipeline"
FEATURES_FILE = OUTPUT_DIR / "lichess_player_features.jsonl"
RAW_GAMES_FILE = OUTPUT_DIR / "lichess_raw_game_data.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "lichess_gather_checkpoint.json"

# Find the zst file
ZST_CANDIDATES = list(LICHESS_DIR.glob("*.pgn.zst")) + list(LICHESS_DIR.glob("*.pgn.zst.crdownload"))

# ---------------------------------------------------------------------------
# MLP (same as elo_pipeline.py)
# ---------------------------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

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
# Load ensemble
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
# Parse eval from Lichess PGN move comments
# ---------------------------------------------------------------------------
EVAL_RE = re.compile(r'\[%eval\s+([#\-\d.]+)\]')

def parse_eval_cp(eval_str):
    """Convert Lichess eval string to centipawns (white POV).
    '#3' -> 10000, '#-2' -> -10000, '1.52' -> 152
    Mate scores capped at +/-10000 to avoid overflow in win% formula.
    """
    if eval_str.startswith('#'):
        mate_num = int(eval_str[1:])
        return 10000 if mate_num > 0 else -10000
    return int(float(eval_str) * 100)


def extract_evals_from_game(game):
    """Walk through a parsed game and extract per-move eval + FEN data.
    Returns list of dicts similar to stockfish_eval_game output, or None if no evals.
    """
    board = game.board()
    node = game
    moves_data = []
    prev_cp_white = None

    for node in game.mainline():
        comment = node.comment or ""
        match = EVAL_RE.search(comment)
        if match is None:
            # If we've been tracking evals and hit a gap, stop
            if prev_cp_white is not None:
                break
            continue

        fen_before = board.fen()
        color = "white" if board.turn == chess.WHITE else "black"
        move_san = board.san(node.move)

        cp_after_white = parse_eval_cp(match.group(1))

        if prev_cp_white is not None:
            # We have before and after — compute accuracy
            if color == "white":
                cp_before = prev_cp_white
                cp_after = cp_after_white
            else:
                cp_before = -prev_cp_white
                cp_after = -cp_after_white

            win_pct_before = get_win_percent(cp_before)
            win_pct_after = get_win_percent(cp_after)
            actual_loss = max(0.0, win_pct_before - win_pct_after)

            moves_data.append({
                "fen_before": fen_before,
                "fen_after": None,  # filled after push
                "move_san": move_san,
                "color": color,
                "cp_before_white": prev_cp_white,
                "cp_after_white": cp_after_white,
                "actual_accuracy_loss": round(actual_loss, 4),
            })

        board.push(node.move)

        if moves_data and moves_data[-1]["fen_after"] is None:
            moves_data[-1]["fen_after"] = board.fen()

        prev_cp_white = cp_after_white

    return moves_data if len(moves_data) >= 10 else None


# ---------------------------------------------------------------------------
# Feature extraction (imported from elo_pipeline)
# ---------------------------------------------------------------------------
from scipy.stats import skew, kurtosis as scipy_kurtosis


def _streak_max(vals, threshold):
    best = cur = 0
    for v in vals:
        if v < threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _safe_slope(values):
    if len(values) < 3:
        return 0.0
    x = np.arange(len(values), dtype=np.float64)
    y = np.array(values, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (y - ym)).sum() / denom)


def extract_player_features(moves_data, ensemble_preds, headers, color):
    player_moves = [(m, e) for m, e in zip(moves_data, ensemble_preds) if m["color"] == color]
    if len(player_moves) < 5:
        return None

    actual_losses = [m["actual_accuracy_loss"] for m, _ in player_moves]
    complexity_scores = [e["complexity"] for _, e in player_moves]
    ensemble_vals = [e["ensemble"] for _, e in player_moves]
    n_moves = len(player_moves)
    losses_arr = np.array(actual_losses)
    complexity_arr = np.array(complexity_scores)
    ensemble_arr = np.array(ensemble_vals)

    mean_actual = float(losses_arr.mean())
    std_actual = float(losses_arr.std())
    median_loss = float(np.median(losses_arr))
    p75_loss = float(np.percentile(losses_arr, 75))
    p90_loss = float(np.percentile(losses_arr, 90))
    max_loss = float(losses_arr.max())

    loss_skewness = float(skew(losses_arr)) if n_moves >= 3 else 0.0
    loss_kurtosis = float(scipy_kurtosis(losses_arr)) if n_moves >= 4 else 0.0
    coeff_variation = std_actual / mean_actual if mean_actual > 0.01 else 0.0
    iqr_loss = float(np.percentile(losses_arr, 75) - np.percentile(losses_arr, 25))

    mean_expected = float(ensemble_arr.mean())
    mean_complexity_score = float(complexity_arr.mean())
    std_complexity = float(complexity_arr.std())
    max_complexity = float(complexity_arr.max())

    ratios, residuals = [], []
    for m, e in player_moves:
        exp = e["ensemble"]
        act = m["actual_accuracy_loss"]
        if exp > 0.01:
            ratios.append(act / (exp * 10))
        residuals.append(act - exp * 10)

    mean_ratio = float(np.mean(ratios)) if ratios else 0
    mean_residual = float(np.mean(residuals))
    std_residual = float(np.std(residuals))
    weights = complexity_arr / complexity_arr.sum() if complexity_arr.sum() > 0 else np.ones(n_moves) / n_moves
    complexity_weighted_loss = float((losses_arr * weights).sum())

    perfect_moves = int((losses_arr < 0.1).sum())
    good_moves = int((losses_arr < 1.0).sum())
    inaccuracies = int(((losses_arr >= 1.0) & (losses_arr < 3.0)).sum())
    mistakes = int(((losses_arr >= 3.0) & (losses_arr < 7.0)).sum())
    blunders = int((losses_arr >= 7.0).sum())
    minor_inaccuracies = int(((losses_arr >= 0.1) & (losses_arr < 0.5)).sum())
    major_blunders = int((losses_arr >= 15.0).sum())

    very_complex = [(m, e) for m, e in player_moves if e["complexity"] >= 75]
    complex_mid = [(m, e) for m, e in player_moves if 50 <= e["complexity"] < 75]
    moderate = [(m, e) for m, e in player_moves if 25 <= e["complexity"] < 50]
    simple = [(m, e) for m, e in player_moves if e["complexity"] < 25]

    mean_loss_very_complex = float(np.mean([m["actual_accuracy_loss"] for m, _ in very_complex])) if very_complex else 0
    mean_loss_complex_mid = float(np.mean([m["actual_accuracy_loss"] for m, _ in complex_mid])) if complex_mid else 0
    mean_loss_moderate = float(np.mean([m["actual_accuracy_loss"] for m, _ in moderate])) if moderate else 0
    mean_loss_simple = float(np.mean([m["actual_accuracy_loss"] for m, _ in simple])) if simple else 0

    pct_perfect_complex = (sum(1 for m, _ in very_complex + complex_mid if m["actual_accuracy_loss"] < 0.1)
                           / max(1, len(very_complex) + len(complex_mid)))
    pct_perfect_simple = (sum(1 for m, _ in simple + moderate if m["actual_accuracy_loss"] < 0.1)
                          / max(1, len(simple) + len(moderate)))
    loss_ratio_complex_simple = (mean_loss_very_complex / mean_loss_simple) if mean_loss_simple > 0.01 else 0

    opening_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if i < 8]
    middle_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if 8 <= i < 20]
    late_losses = [m["actual_accuracy_loss"] for i, (m, _) in enumerate(player_moves) if i >= 20]

    mean_loss_opening = float(np.mean(opening_losses)) if opening_losses else 0
    mean_loss_middlegame = float(np.mean(middle_losses)) if middle_losses else 0
    mean_loss_endgame = float(np.mean(late_losses)) if late_losses else 0
    pct_perfect_opening = (sum(1 for l in opening_losses if l < 0.1) / max(1, len(opening_losses)))
    pct_perfect_endgame = (sum(1 for l in late_losses if l < 0.1) / max(1, len(late_losses)))

    accuracy_slope = _safe_slope(actual_losses)
    complexity_slope = _safe_slope(complexity_scores)

    behind_moves, ahead_moves, equal_moves = [], [], []
    for m, e in player_moves:
        cp = m["cp_before_white"]
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

    longest_good_streak = _streak_max(actual_losses, 1.0)
    longest_perfect_streak = _streak_max(actual_losses, 0.1)
    good_streak_ratio = longest_good_streak / n_moves
    perfect_streak_ratio = longest_perfect_streak / n_moves
    consecutive_errors = sum(1 for i in range(1, n_moves) if actual_losses[i] > 2.0 and actual_losses[i-1] > 2.0)

    perfect_times_complexity = (perfect_moves / n_moves) * mean_complexity_score
    blunder_in_simple = sum(1 for m, e in player_moves if m["actual_accuracy_loss"] >= 7.0 and e["complexity"] < 50)
    blunder_in_simple_rate = blunder_in_simple / max(1, len(simple) + len(moderate))

    eval_changes = []
    for m, _ in player_moves:
        cp_before = m["cp_before_white"] if color == "white" else -m["cp_before_white"]
        cp_after = m["cp_after_white"] if color == "white" else -m["cp_after_white"]
        eval_changes.append(cp_after - cp_before)
    mean_eval_change = float(np.mean(eval_changes))
    std_eval_change = float(np.std(eval_changes))

    elo_key = "WhiteElo" if color == "white" else "BlackElo"
    try:
        elo = int(headers.get(elo_key))
    except (TypeError, ValueError):
        return None

    return {
        "elo": elo,
        "color": color,
        "player": headers.get("White" if color == "white" else "Black", "?"),
        "opponent_elo": _safe_int(headers.get("BlackElo" if color == "white" else "WhiteElo")),
        "result": headers.get("Result", "?"),
        "time_control": headers.get("TimeControl", "?"),
        "source": "lichess",
        "n_moves": n_moves,
        "game_length": len(moves_data),
        "mean_actual_loss": round(mean_actual, 4),
        "std_actual_loss": round(std_actual, 4),
        "median_loss": round(median_loss, 4),
        "p75_loss": round(p75_loss, 4),
        "p90_loss": round(p90_loss, 4),
        "max_loss": round(max_loss, 4),
        "loss_skewness": round(loss_skewness, 4),
        "loss_kurtosis": round(loss_kurtosis, 4),
        "coeff_variation": round(coeff_variation, 4),
        "iqr_loss": round(iqr_loss, 4),
        "mean_expected_ensemble": round(mean_expected, 4),
        "mean_complexity_score": round(mean_complexity_score, 2),
        "std_complexity": round(std_complexity, 2),
        "max_complexity": round(max_complexity, 2),
        "mean_ratio": round(mean_ratio, 4),
        "mean_residual": round(mean_residual, 4),
        "std_residual": round(std_residual, 4),
        "complexity_weighted_loss": round(complexity_weighted_loss, 4),
        "pct_perfect": round(perfect_moves / n_moves, 4),
        "pct_good": round(good_moves / n_moves, 4),
        "pct_inaccuracy": round(inaccuracies / n_moves, 4),
        "pct_mistake": round(mistakes / n_moves, 4),
        "pct_blunder": round(blunders / n_moves, 4),
        "pct_minor_inaccuracy": round(minor_inaccuracies / n_moves, 4),
        "pct_major_blunder": round(major_blunders / n_moves, 4),
        "mean_loss_very_complex": round(mean_loss_very_complex, 4),
        "mean_loss_complex_mid": round(mean_loss_complex_mid, 4),
        "mean_loss_moderate": round(mean_loss_moderate, 4),
        "mean_loss_simple": round(mean_loss_simple, 4),
        "n_very_complex": len(very_complex),
        "n_complex_mid": len(complex_mid),
        "n_simple": len(simple),
        "pct_perfect_complex": round(pct_perfect_complex, 4),
        "pct_perfect_simple": round(pct_perfect_simple, 4),
        "loss_ratio_complex_simple": round(loss_ratio_complex_simple, 4),
        "mean_loss_opening": round(mean_loss_opening, 4),
        "mean_loss_middlegame": round(mean_loss_middlegame, 4),
        "mean_loss_endgame": round(mean_loss_endgame, 4),
        "pct_perfect_opening": round(pct_perfect_opening, 4),
        "pct_perfect_endgame": round(pct_perfect_endgame, 4),
        "accuracy_slope": round(accuracy_slope, 6),
        "complexity_slope": round(complexity_slope, 4),
        "mean_loss_behind": round(mean_loss_behind, 4),
        "mean_loss_ahead": round(mean_loss_ahead, 4),
        "mean_loss_equal": round(mean_loss_equal, 4),
        "pct_moves_behind": round(pct_moves_behind, 4),
        "pct_moves_ahead": round(pct_moves_ahead, 4),
        "longest_good_streak": longest_good_streak,
        "longest_perfect_streak": longest_perfect_streak,
        "good_streak_ratio": round(good_streak_ratio, 4),
        "perfect_streak_ratio": round(perfect_streak_ratio, 4),
        "consecutive_errors": consecutive_errors,
        "perfect_times_complexity": round(perfect_times_complexity, 4),
        "blunder_in_simple_rate": round(blunder_in_simple_rate, 4),
        "mean_eval_change": round(mean_eval_change, 2),
        "std_eval_change": round(std_eval_change, 2),
    }


def _safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Main gather loop
# ---------------------------------------------------------------------------
def run(time_limit_minutes=5):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find zst file
    zst_files = list(LICHESS_DIR.glob("*.pgn.zst")) + list(LICHESS_DIR.glob("*.pgn.zst.crdownload"))
    if not zst_files:
        print("No .pgn.zst file found in", LICHESS_DIR)
        return
    zst_path = zst_files[0]
    print(f"Reading: {zst_path.name}")

    # Load ensemble
    cnn_sd, attn_cnn, mlp, cal, device = load_ensemble()

    # Resume
    games_offset = 0
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        games_offset = ckpt.get("games_scanned", 0)
        print(f"Resuming from game {games_offset}")

    out_f = open(FEATURES_FILE, "a")
    raw_f = open(RAW_GAMES_FILE, "a")
    t0 = time.time()
    deadline = t0 + time_limit_minutes * 60

    games_scanned = 0
    games_with_eval = 0
    features_written = 0
    skipped_resume = 0
    last_report = t0

    print(f"Gathering Lichess data — {time_limit_minutes} min limit")

    dctx = zstd.ZstdDecompressor()
    with open(zst_path, "rb") as f:
        reader = dctx.stream_reader(f)
        text_stream = io.TextIOWrapper(reader, encoding="utf-8")
        pgn_io = text_stream  # chess.pgn can read from any text stream

        while True:
            if time.time() > deadline:
                print("Time limit reached.")
                break

            game = chess.pgn.read_game(pgn_io)
            if game is None:
                print("End of file.")
                break

            games_scanned += 1

            # Skip already-processed games on resume
            if skipped_resume < games_offset:
                skipped_resume += 1
                continue

            # Check if game has evals (quick check on first node)
            has_eval = False
            for node in game.mainline():
                if node.comment and "%eval" in node.comment:
                    has_eval = True
                break
            if not has_eval:
                continue

            # Extract move data from embedded evals
            moves_data = extract_evals_from_game(game)
            if moves_data is None:
                continue

            games_with_eval += 1
            headers = dict(game.headers)

            # Ensemble predictions
            before_fens = [m["fen_before"] for m in moves_data]
            ensemble_preds = batch_ensemble_predict(
                before_fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=512
            )

            # Save raw data
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
            raw_f.write(json.dumps({"headers": headers, "moves": raw_moves}) + "\n")

            # Extract features for both players
            for color in ["white", "black"]:
                features = extract_player_features(moves_data, ensemble_preds, headers, color)
                if features is not None:
                    out_f.write(json.dumps(features) + "\n")
                    features_written += 1

            # Progress report every 15s
            now = time.time()
            if now - last_report > 15:
                elapsed = now - t0
                remaining = (deadline - now) / 60
                rate_games = games_scanned / elapsed
                rate_eval = games_with_eval / elapsed * 60
                print(f"  [{elapsed/60:.1f}m] Scanned: {games_scanned:,} | "
                      f"With eval: {games_with_eval:,} | "
                      f"Features: {features_written:,} | "
                      f"{rate_eval:.0f} eval-games/min | "
                      f"{remaining:.1f}m left")
                last_report = now
                out_f.flush()
                raw_f.flush()

                with open(CHECKPOINT_FILE, "w") as cf:
                    json.dump({
                        "games_scanned": games_scanned,
                        "games_with_eval": games_with_eval,
                        "features_written": features_written,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }, cf, indent=2)

    out_f.close()
    raw_f.close()
    elapsed = time.time() - t0

    with open(CHECKPOINT_FILE, "w") as cf:
        json.dump({
            "games_scanned": games_scanned,
            "games_with_eval": games_with_eval,
            "features_written": features_written,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, cf, indent=2)

    print()
    print("=" * 60)
    print(f"LICHESS GATHER COMPLETE")
    print(f"  Games scanned:    {games_scanned:,}")
    print(f"  Games with eval:  {games_with_eval:,} ({games_with_eval/max(1,games_scanned)*100:.1f}%)")
    print(f"  Features written: {features_written:,}")
    print(f"  Elapsed:          {elapsed/60:.1f} min")
    print(f"  Rate:             {games_with_eval/elapsed*60:.0f} eval-games/min")
    print(f"  Output:           {FEATURES_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=5, help="Time limit in minutes")
    args = parser.parse_args()
    run(time_limit_minutes=args.time)
