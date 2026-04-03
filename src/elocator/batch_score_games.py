"""Batch score every game in filtered.pgn with the 3-model ensemble.

For each game, stores:
  - Game headers (White, Black, Event, Result, Elo, etc.)
  - Per-position: FEN, raw predictions from each model, ensemble score, complexity score
  - Game-level summary stats

Checkpoints every N games. Fully resumable.
"""

import sys
import os
import time
import json
import bisect
import signal
from pathlib import Path
from collections import defaultdict

import chess
import chess.pgn
import torch
import numpy as np

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import fen_to_tensor, fen_encoder
from model_cnn import ChessCNNModel, AttentionCNN

# ---------------------------------------------------------------------------
# MLP architecture (copied from app.py to avoid FastAPI import)
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
# Config
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "api" / "model"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
PGN_PATH = DATA_DIR / "filtered.pgn"
OUTPUT_DIR = DATA_DIR / "game_scores"
OUTPUT_FILE = OUTPUT_DIR / "scored_games.jsonl"
CHECKPOINT_FILE = OUTPUT_DIR / "checkpoint.json"
BATCH_PROFILE_FILE = OUTPUT_DIR / "batch_profile.json"

SAVE_EVERY_N_GAMES = 500  # checkpoint interval


# ---------------------------------------------------------------------------
# Load models & calibration
# ---------------------------------------------------------------------------
def load_models():
    device = "cpu"
    if torch.backends.mps.is_available():
        try:
            torch.tensor([], device="mps")
            device = "mps"
        except RuntimeError:
            pass
    print(f"Using device: {device}")

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

    return cnn_sd, attn_cnn, mlp, cal, device


def _minmax_norm(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5


# ---------------------------------------------------------------------------
# Batch encoding
# ---------------------------------------------------------------------------
def batch_encode_fens(fens):
    """Encode a list of FENs into batched CNN and MLP tensors."""
    cnn_tensors = []
    mlp_vectors = []
    for fen in fens:
        cnn_tensors.append(fen_to_tensor(fen))
        mlp_vectors.append(fen_encoder(fen))
    cnn_batch = torch.stack(cnn_tensors)
    mlp_batch = torch.tensor(mlp_vectors, dtype=torch.float32)
    return cnn_batch, mlp_batch


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------
@torch.no_grad()
def batch_predict(fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=256):
    """Score a list of FENs through all 3 models. Returns per-position dicts."""
    sd_min, sd_max = cal["sd_min"], cal["sd_max"]
    attn_min, attn_max = cal["attn_min"], cal["attn_max"]
    mlp_min, mlp_max = cal["mlp_min"], cal["mlp_max"]
    breakpoints = cal["breakpoints"]

    all_results = []
    for i in range(0, len(fens), batch_size):
        chunk = fens[i:i + batch_size]
        cnn_batch, mlp_batch = batch_encode_fens(chunk)
        cnn_batch = cnn_batch.to(device)
        mlp_batch = mlp_batch.to(device)

        sd_preds = cnn_sd(cnn_batch).squeeze(1).cpu().numpy()
        attn_preds = attn_cnn(cnn_batch).squeeze(1).cpu().numpy()
        mlp_preds = mlp(mlp_batch).squeeze(1).cpu().numpy() * 100  # sigmoid → raw

        for j in range(len(chunk)):
            sd_raw = float(sd_preds[j])
            attn_raw = float(attn_preds[j])
            mlp_raw = float(mlp_preds[j])

            sd_norm = _minmax_norm(sd_raw, sd_min, sd_max)
            attn_norm = _minmax_norm(attn_raw, attn_min, attn_max)
            mlp_norm = _minmax_norm(mlp_raw, mlp_min, mlp_max)
            ensemble = (sd_norm + attn_norm + mlp_norm) / 3

            complexity = max(1, min(100, bisect.bisect_left(breakpoints, ensemble) + 1))

            all_results.append({
                "fen": chunk[j],
                "sd_raw": round(sd_raw, 4),
                "attn_raw": round(attn_raw, 4),
                "mlp_raw": round(mlp_raw, 4),
                "sd_norm": round(sd_norm, 4),
                "attn_norm": round(attn_norm, 4),
                "mlp_norm": round(mlp_norm, 4),
                "ensemble": round(ensemble, 4),
                "complexity": complexity,
            })
    return all_results


# ---------------------------------------------------------------------------
# PGN parsing — extract game headers + all positions
# ---------------------------------------------------------------------------
def parse_game_positions(game):
    """Extract headers and list of FENs from a chess.pgn.Game."""
    headers = dict(game.headers)
    fens = []
    board = game.board()
    fens.append(board.fen())
    for move in game.mainline_moves():
        board.push(move)
        fens.append(board.fen())
    return headers, fens


# ---------------------------------------------------------------------------
# Batch size profiler
# ---------------------------------------------------------------------------
def profile_batch_sizes(cnn_sd, attn_cnn, mlp, cal, device):
    """Test different batch sizes and find the optimal one."""
    # Generate some test FENs by parsing a few games
    print("Profiling batch sizes...")
    test_fens = []
    with open(PGN_PATH) as f:
        for _ in range(5):
            game = chess.pgn.read_game(f)
            if game is None:
                break
            _, fens = parse_game_positions(game)
            test_fens.extend(fens)

    # Pad to at least 1024 positions by repeating
    while len(test_fens) < 1024:
        test_fens.extend(test_fens[:1024 - len(test_fens)])
    test_fens = test_fens[:1024]

    results = {}
    for bs in [32, 64, 128, 256, 512, 1024]:
        # Warmup
        batch_predict(test_fens[:bs], cnn_sd, attn_cnn, mlp, cal, device, batch_size=bs)

        t0 = time.time()
        n_iters = 3
        for _ in range(n_iters):
            batch_predict(test_fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=bs)
        elapsed = (time.time() - t0) / n_iters
        pos_per_sec = len(test_fens) / elapsed
        results[bs] = {"elapsed_s": round(elapsed, 3), "positions_per_sec": round(pos_per_sec, 1)}
        print(f"  batch_size={bs:4d}: {pos_per_sec:.0f} pos/s ({elapsed:.3f}s for {len(test_fens)} positions)")

    best_bs = max(results, key=lambda k: results[k]["positions_per_sec"])
    print(f"  => Best batch size: {best_bs} ({results[best_bs]['positions_per_sec']:.0f} pos/s)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(BATCH_PROFILE_FILE, "w") as f:
        json.dump({"results": results, "best": best_bs}, f, indent=2)

    return best_bs, results


# ---------------------------------------------------------------------------
# Main scoring loop
# ---------------------------------------------------------------------------
STOP_REQUESTED = False

def handle_signal(signum, frame):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    print("\n[SIGINT] Graceful stop requested, finishing current batch...")


def load_checkpoint():
    """Load checkpoint: returns (games_completed, byte_offset)."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        return ckpt["games_completed"], ckpt.get("byte_offset", 0)
    return 0, 0


def save_checkpoint(games_completed, byte_offset, stats):
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "games_completed": games_completed,
            "byte_offset": byte_offset,
            "stats": stats,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)


def run(time_limit_minutes=30):
    global STOP_REQUESTED
    signal.signal(signal.SIGINT, handle_signal)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load models
    cnn_sd, attn_cnn, mlp, cal, device = load_models()

    # Profile batch sizes
    if BATCH_PROFILE_FILE.exists():
        with open(BATCH_PROFILE_FILE) as f:
            profile = json.load(f)
        batch_size = profile["best"]
        print(f"Using cached batch size: {batch_size}")
    else:
        batch_size, _ = profile_batch_sizes(cnn_sd, attn_cnn, mlp, cal, device)

    # Resume from checkpoint
    games_completed, byte_offset = load_checkpoint()
    if games_completed > 0:
        print(f"Resuming from game {games_completed} (byte offset {byte_offset})")

    # Open output file in append mode
    out_f = open(OUTPUT_FILE, "a")

    # Stats tracking
    stats = {
        "total_games": 0,
        "total_positions": 0,
        "total_time_s": 0,
    }

    start_time = time.time()
    deadline = start_time + time_limit_minutes * 60
    games_this_session = 0
    positions_this_session = 0
    last_save_game_count = games_completed
    last_report_time = start_time

    print(f"Starting scoring run — {time_limit_minutes} min limit, batch_size={batch_size}")
    print(f"PGN: {PGN_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print()

    with open(PGN_PATH) as pgn_f:
        if byte_offset > 0:
            pgn_f.seek(byte_offset)

        while True:
            if STOP_REQUESTED or time.time() > deadline:
                break

            # Read next game
            game = chess.pgn.read_game(pgn_f)
            if game is None:
                print("Reached end of PGN file!")
                break

            current_offset = pgn_f.tell()

            # Parse positions
            try:
                headers, fens = parse_game_positions(game)
            except Exception as e:
                games_completed += 1
                continue

            if len(fens) < 2:
                games_completed += 1
                continue

            # Batch score all positions in this game
            position_scores = batch_predict(fens, cnn_sd, attn_cnn, mlp, cal, device, batch_size=batch_size)

            # Compute game-level summary
            complexities = [p["complexity"] for p in position_scores]
            ensembles = [p["ensemble"] for p in position_scores]
            game_record = {
                "game_index": games_completed,
                "headers": headers,
                "num_positions": len(fens),
                "summary": {
                    "mean_complexity": round(np.mean(complexities), 2),
                    "max_complexity": max(complexities),
                    "min_complexity": min(complexities),
                    "median_complexity": round(float(np.median(complexities)), 2),
                    "std_complexity": round(float(np.std(complexities)), 2),
                    "mean_ensemble": round(np.mean(ensembles), 4),
                    "max_ensemble": round(max(ensembles), 4),
                    "positions_above_90": sum(1 for c in complexities if c >= 90),
                    "positions_above_75": sum(1 for c in complexities if c >= 75),
                },
                "positions": position_scores,
            }

            out_f.write(json.dumps(game_record) + "\n")

            games_completed += 1
            games_this_session += 1
            positions_this_session += len(fens)

            # Periodic checkpoint
            if games_completed - last_save_game_count >= SAVE_EVERY_N_GAMES:
                out_f.flush()
                elapsed = time.time() - start_time
                stats = {
                    "total_games_scored": games_completed,
                    "session_games": games_this_session,
                    "session_positions": positions_this_session,
                    "session_elapsed_s": round(elapsed, 1),
                    "positions_per_sec": round(positions_this_session / elapsed, 1) if elapsed > 0 else 0,
                    "games_per_min": round(games_this_session / (elapsed / 60), 1) if elapsed > 0 else 0,
                }
                save_checkpoint(games_completed, current_offset, stats)
                last_save_game_count = games_completed

            # Progress report every 30s
            now = time.time()
            if now - last_report_time > 30:
                elapsed = now - start_time
                remaining = deadline - now
                pos_per_sec = positions_this_session / elapsed if elapsed > 0 else 0
                games_per_min = games_this_session / (elapsed / 60) if elapsed > 0 else 0
                print(f"[{elapsed/60:.1f}m] Games: {games_completed} (+{games_this_session}) | "
                      f"Positions: {positions_this_session:,} | "
                      f"{pos_per_sec:.0f} pos/s | {games_per_min:.1f} games/min | "
                      f"{remaining/60:.1f}m remaining")
                last_report_time = now

    # Final save
    out_f.flush()
    out_f.close()
    elapsed = time.time() - start_time
    stats = {
        "total_games_scored": games_completed,
        "session_games": games_this_session,
        "session_positions": positions_this_session,
        "session_elapsed_s": round(elapsed, 1),
        "positions_per_sec": round(positions_this_session / elapsed, 1) if elapsed > 0 else 0,
        "games_per_min": round(games_this_session / (elapsed / 60), 1) if elapsed > 0 else 0,
    }
    save_checkpoint(games_completed, 0, stats)

    print()
    print("=" * 60)
    print(f"SESSION COMPLETE")
    print(f"  Games scored this session: {games_this_session}")
    print(f"  Total games scored:        {games_completed}")
    print(f"  Positions scored:          {positions_this_session:,}")
    print(f"  Elapsed:                   {elapsed/60:.1f} minutes")
    print(f"  Throughput:                {stats['positions_per_sec']} pos/s, {stats['games_per_min']} games/min")
    print(f"  Output:                    {OUTPUT_FILE}")
    print("=" * 60)
    return stats


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", type=float, default=30, help="Time limit in minutes")
    parser.add_argument("--reprofile", action="store_true", help="Re-run batch size profiling")
    args = parser.parse_args()

    if args.reprofile and BATCH_PROFILE_FILE.exists():
        BATCH_PROFILE_FILE.unlink()

    run(time_limit_minutes=args.time)
