"""Sample one random move from randomly-ordered games and evaluate with Stockfish.

Produces JSONL training data: {"fen": ..., "accuracy": ...}

Design:
  - Two passes: (1) index all game byte-offsets, (2) visit in shuffled order
  - Deterministic: seeded shuffle + seeded move selection = fully reproducible
  - Resumable: checkpoints progress every N records so it can be stopped and restarted
  - Two Stockfish evals per game: position before move + position after move
  - Position cache avoids redundant Stockfish calls across games
"""

import chess
import chess.engine
import chess.pgn
import json
import os
import random
import argparse
import time

from utils import calculate_win_percentage_change

# Defaults
DEFAULT_DEPTH = 18
DEFAULT_THREADS = 8
DEFAULT_CHECKPOINT_INTERVAL = 1000


def build_game_index(pgn_path, index_path):
    """First pass: record byte offset of every game in the PGN file.

    Saves the index to disk so it only needs to be done once.
    Returns list of byte offsets.
    """
    if os.path.exists(index_path):
        print(f"Loading existing game index from {index_path}...")
        with open(index_path, 'r') as f:
            offsets = json.load(f)
        print(f"  {len(offsets):,} games indexed")
        return offsets

    print(f"Building game index for {pgn_path} (first run only)...")
    offsets = []
    with open(pgn_path, errors='replace') as f:
        while True:
            offset = f.tell()
            game = chess.pgn.read_game(f)
            if game is None:
                break
            offsets.append(offset)
            if len(offsets) % 100000 == 0:
                print(f"  Indexed {len(offsets):,} games...", flush=True)

    print(f"  Done: {len(offsets):,} games indexed")
    with open(index_path, 'w') as f:
        json.dump(offsets, f)
    return offsets


def load_checkpoint(checkpoint_path):
    """Load checkpoint: returns number of games already processed."""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            data = json.load(f)
        return data.get('games_processed', 0)
    return 0


def save_checkpoint(checkpoint_path, games_processed):
    """Save checkpoint with current progress."""
    with open(checkpoint_path, 'w') as f:
        json.dump({'games_processed': games_processed}, f)


def load_position_cache(cache_path):
    """Load the FEN -> score cache."""
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}


def save_position_cache(cache_path, cache):
    """Save the FEN -> score cache."""
    with open(cache_path, 'w') as f:
        json.dump(cache, f)


def eval_position(board, engine, depth, time_limit, cache):
    """Evaluate a position, using cache if available. Returns centipawn score from white's POV.

    Limit behavior:
      --time only:         each eval gets exactly that many seconds
      --depth only:        search to exact depth (can be slow on complex positions)
      --depth AND --time:  search to depth OR time cap, whichever comes first
    """
    fen = board.fen()
    if fen in cache:
        return cache[fen]
    if time_limit and depth != DEFAULT_DEPTH:
        # Both specified: cap at whichever comes first
        limit = chess.engine.Limit(depth=depth, time=time_limit)
    elif time_limit:
        limit = chess.engine.Limit(time=time_limit)
    else:
        limit = chess.engine.Limit(depth=depth)
    info = engine.analyse(board, limit)
    score = info["score"].white().score(mate_score=100000)
    cache[fen] = score
    return score


def sample_and_eval(
    pgn_path,
    output_path,
    checkpoint_path,
    cache_path,
    index_path,
    depth=DEFAULT_DEPTH,
    time_limit=None,
    threads=DEFAULT_THREADS,
    checkpoint_interval=DEFAULT_CHECKPOINT_INTERVAL,
    seed=42,
    max_games=None,
):
    """Sample one random move per game in shuffled order, evaluate with Stockfish."""

    # Pass 1: build/load game index
    offsets = build_game_index(pgn_path, index_path)

    # Shuffle deterministically
    rng = random.Random(seed)
    shuffled_order = list(range(len(offsets)))
    rng.shuffle(shuffled_order)

    # Resume
    games_processed = load_checkpoint(checkpoint_path)
    cache = load_position_cache(cache_path)
    mode = f"time={time_limit}s/eval" if time_limit else f"depth={depth}"
    print(f"Mode: {mode}, threads={threads}")
    print(f"Resuming from position {games_processed:,}/{len(shuffled_order):,}, "
          f"cache has {len(cache):,} positions")

    if max_games:
        end = min(games_processed + max_games, len(shuffled_order))
    else:
        end = len(shuffled_order)

    records_written = 0
    batch = []
    start_time = time.time()

    with open(pgn_path, errors='replace') as pgn_file, \
         chess.engine.SimpleEngine.popen_uci("/opt/homebrew/bin/stockfish") as engine:

        engine.configure({"Threads": threads})

        for seq_idx in range(games_processed, end):
            game_idx = shuffled_order[seq_idx]
            byte_offset = offsets[game_idx]

            # Seek to this game
            pgn_file.seek(byte_offset)
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                continue

            # Get all moves
            moves = list(game.mainline_moves())
            if len(moves) < 10:
                continue

            # Deterministic random move selection: seed based on original game index
            move_rng = random.Random(seed + game_idx)
            # Avoid first 4 half-moves (book moves) and last 2 (often trivial)
            lo = min(4, len(moves) - 1)
            hi = max(lo + 1, len(moves) - 2)
            move_idx = move_rng.randint(lo, hi)

            # Replay to the chosen move
            board = game.board()
            for i, move in enumerate(moves):
                if i == move_idx:
                    fen_before = board.fen()
                    player_perspective = board.turn

                    # Eval before
                    score_before_white = eval_position(board, engine, depth, time_limit, cache)
                    score_before = score_before_white if player_perspective == chess.WHITE else -score_before_white

                    # Make the move
                    board.push(move)

                    # Eval after
                    score_after_white = eval_position(board, engine, depth, time_limit, cache)
                    score_after = score_after_white if player_perspective == chess.WHITE else -score_after_white

                    # Accuracy
                    accuracy = calculate_win_percentage_change(score_before, score_after)

                    batch.append({
                        "fen": fen_before,
                        "accuracy": round(accuracy, 4),
                    })
                    records_written += 1
                    break
                board.push(move)

            # Checkpoint
            count = seq_idx - games_processed + 1
            if count % checkpoint_interval == 0:
                if batch:
                    with open(output_path, 'a') as f:
                        for record in batch:
                            f.write(json.dumps(record) + '\n')
                    batch = []

                save_checkpoint(checkpoint_path, seq_idx + 1)
                save_position_cache(cache_path, cache)

                elapsed = time.time() - start_time
                rate = records_written / elapsed if elapsed > 0 else 0
                print(f"Progress: {seq_idx + 1:,}/{end:,} | Written: {records_written:,} | "
                      f"Cache: {len(cache):,} | Rate: {rate:.1f} rec/s | "
                      f"Elapsed: {elapsed/60:.1f}m", flush=True)

    # Final flush
    if batch:
        with open(output_path, 'a') as f:
            for record in batch:
                f.write(json.dumps(record) + '\n')
    save_checkpoint(checkpoint_path, end)
    save_position_cache(cache_path, cache)

    elapsed = time.time() - start_time
    print(f"\nDone. Processed {records_written:,} games in {elapsed/60:.1f} minutes "
          f"({elapsed/3600:.1f} hours)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample & evaluate one move per game")
    parser.add_argument("--input", default="./data/filtered.pgn",
                        help="Input filtered PGN file")
    parser.add_argument("--output", default="./data/train_sampled.jsonl",
                        help="Output JSONL file (appended)")
    parser.add_argument("--checkpoint", default="./data/sample_checkpoint.json",
                        help="Checkpoint file for resume")
    parser.add_argument("--cache", default="./data/position_cache.json",
                        help="Position eval cache file")
    parser.add_argument("--index", default="./data/game_index.json",
                        help="Game byte-offset index file")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH,
                        help=f"Stockfish search depth (default: {DEFAULT_DEPTH}). Ignored if --time is set.")
    parser.add_argument("--time", type=float, default=None,
                        help="Seconds per Stockfish eval (e.g. 0.1). Overrides --depth. Engine searches as deep as it can in the time given.")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                        help=f"Stockfish threads (default: {DEFAULT_THREADS})")
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL,
                        help=f"Save every N games (default: {DEFAULT_CHECKPOINT_INTERVAL})")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for deterministic shuffle + move selection")
    parser.add_argument("--max-games", type=int, default=None,
                        help="Stop after this many new games")
    args = parser.parse_args()

    sample_and_eval(
        pgn_path=args.input,
        output_path=args.output,
        checkpoint_path=args.checkpoint,
        cache_path=args.cache,
        index_path=args.index,
        depth=args.depth,
        time_limit=args.time,
        threads=args.threads,
        checkpoint_interval=args.checkpoint_interval,
        seed=args.seed,
        max_games=args.max_games,
    )
