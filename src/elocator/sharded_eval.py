#!/usr/bin/env python3
"""Sharded Stockfish evaluation across EC2 spot instances.

Splits games into N shards, launches 192-core spot workers to evaluate
one random move per game at D20 (with time cap), collects results via S3.

Architecture:
  1. Master builds/loads game index, splits into N shard ranges
  2. Master uploads filtered.pgn, game_index.json, and worker script to S3
  3. Workers install Stockfish, download artifacts, evaluate their shard
  4. Workers upload results to S3 every 1000 positions + on completion
  5. Master polls S3 for done markers, downloads and merges results

Usage:
    cd ~/dev/elocator && poetry run python src/elocator/sharded_eval.py \
        --pgn ./data/filtered.pgn \
        --index ./data/game_index.json \
        --output ./data/eval_d20_t10s/train.jsonl \
        --num-shards 4 \
        --s3-prefix s3://elocator-data/eval_d20_t10s \
        --depth 20 --time 10
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "bg" / "train"))
from ec2_fleet import (
    FleetConfig,
    WorkerState,
    check_worker_health,
    cleanup_stale_workers,
    launch_fleet,
    relaunch_worker,
    terminate_fleet,
    log,
)


S3_BUCKET = "elocator-data"


# ---------------------------------------------------------------------------
# S3 helpers (same pattern as bg/train/sharded_label.py)
# ---------------------------------------------------------------------------

def s3_cp(local_path, s3_path, quiet=True):
    cmd = ["aws", "s3", "cp", local_path, s3_path]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {result.stderr[:500]}")


def s3_download(s3_path, local_path, quiet=True):
    cmd = ["aws", "s3", "cp", s3_path, local_path]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"S3 download failed: {result.stderr[:500]}")


def s3_exists(s3_path):
    result = subprocess.run(
        ["aws", "s3", "ls", s3_path], capture_output=True, text=True
    )
    return result.returncode == 0 and len(result.stdout.strip()) > 0


def s3_head(s3_path):
    parts = s3_path.replace("s3://", "").split("/", 1)
    if len(parts) != 2:
        return None
    bucket, key = parts
    result = subprocess.run(
        ["aws", "s3api", "head-object", "--bucket", bucket, "--key", key,
         "--query", "ContentLength", "--output", "text"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Shard assignment
# ---------------------------------------------------------------------------

def build_shard_ranges(total_games, num_shards):
    """Split game indices into N contiguous ranges for the shuffled order.

    Returns list of (start, end) tuples into the shuffled_order array.
    """
    shard_size = total_games // num_shards
    remainder = total_games % num_shards
    ranges = []
    offset = 0
    for i in range(num_shards):
        count = shard_size + (1 if i < remainder else 0)
        ranges.append((offset, offset + count))
        offset += count
    return ranges


# ---------------------------------------------------------------------------
# Worker script (runs on EC2)
# ---------------------------------------------------------------------------

def build_worker_script(s3_work, shard_id, shard_start, shard_end,
                        depth, time_limit, seed, threads_per_position):
    """Build the bash user-data script for a worker instance."""

    # The worker Python script is embedded inline to avoid uploading a separate file
    worker_py = r'''
import chess
import chess.engine
import chess.pgn
import json
import os
import random
import time
import subprocess
import sys
from multiprocessing import Pool, cpu_count

SHARD_ID = int(os.environ["SHARD_ID"])
SHARD_START = int(os.environ["SHARD_START"])
SHARD_END = int(os.environ["SHARD_END"])
DEPTH = int(os.environ.get("DEPTH", "20"))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", "10"))
SEED = int(os.environ.get("SEED", "42"))
THREADS_PER_POS = int(os.environ.get("THREADS_PER_POS", "8"))
S3_WORK = os.environ["S3_WORK"]
CHECKPOINT_INTERVAL = 1000

def get_win_percent(centipawns):
    return 50 + 50 * (2 / (1 + 10 ** (-centipawns / 400)) - 1)

def calculate_win_percentage_change(cp_before, cp_after):
    wp_before = get_win_percent(cp_before)
    wp_after = get_win_percent(cp_after)
    return max(0, -(wp_after - wp_before))


def eval_one_game(args):
    """Evaluate one game in a subprocess with its own Stockfish instance."""
    game_idx, byte_offset, seed, depth, time_limit, threads = args
    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        engine.configure({"Threads": threads})

        with open("/opt/elocator/filtered.pgn", errors="replace") as f:
            f.seek(byte_offset)
            game = chess.pgn.read_game(f)

        if game is None:
            engine.quit()
            return None

        moves = list(game.mainline_moves())
        if len(moves) < 10:
            engine.quit()
            return None

        rng = random.Random(seed + game_idx)
        lo = min(4, len(moves) - 1)
        hi = max(lo + 1, len(moves) - 2)
        move_idx = rng.randint(lo, hi)

        board = game.board()
        for i, move in enumerate(moves):
            if i == move_idx:
                fen_before = board.fen()
                player = board.turn

                if time_limit > 0 and depth > 0:
                    limit = chess.engine.Limit(depth=depth, time=time_limit)
                elif time_limit > 0:
                    limit = chess.engine.Limit(time=time_limit)
                else:
                    limit = chess.engine.Limit(depth=depth)

                info_before = engine.analyse(board, limit)
                sb_w = info_before["score"].white().score(mate_score=100000)
                sb = sb_w if player == chess.WHITE else -sb_w

                board.push(move)

                info_after = engine.analyse(board, limit)
                sa_w = info_after["score"].white().score(mate_score=100000)
                sa = sa_w if player == chess.WHITE else -sa_w

                accuracy = calculate_win_percentage_change(sb, sa)
                engine.quit()
                return {"fen": fen_before, "accuracy": round(accuracy, 4)}
            board.push(move)

        engine.quit()
        return None
    except Exception as e:
        print(f"Error on game {game_idx}: {e}", flush=True)
        return None


def main():
    print(f"Worker shard {SHARD_ID}: games {SHARD_START}-{SHARD_END}", flush=True)

    # Load game index and shuffled order
    with open("/opt/elocator/game_index.json") as f:
        offsets = json.load(f)

    rng = random.Random(SEED)
    shuffled_order = list(range(len(offsets)))
    rng.shuffle(shuffled_order)

    # Our slice of the shuffled order
    my_games = shuffled_order[SHARD_START:SHARD_END]
    total = len(my_games)
    print(f"  {total} games to evaluate", flush=True)

    # Determine parallelism: N simultaneous Stockfish instances, each with THREADS_PER_POS threads
    num_cpus = cpu_count()
    num_workers = max(1, num_cpus // THREADS_PER_POS)
    print(f"  {num_cpus} CPUs, {num_workers} parallel workers x {THREADS_PER_POS} threads each", flush=True)

    # Build work items
    work_items = [
        (game_idx, offsets[game_idx], SEED, DEPTH, TIME_LIMIT, THREADS_PER_POS)
        for game_idx in my_games
    ]

    output_path = "/opt/elocator/results.jsonl"
    checkpoint_path = "/opt/elocator/results.checkpoint.jsonl"
    records = []
    completed = 0
    t0 = time.time()

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(eval_one_game, work_items, chunksize=10):
            if result is not None:
                records.append(result)
            completed += 1

            if completed % CHECKPOINT_INTERVAL == 0:
                # Write checkpoint
                with open(checkpoint_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (total - completed) / rate if rate > 0 else 0
                print(f"  [{completed}/{total}] {len(records)} records, "
                      f"{rate:.1f} pos/s, ETA {eta/60:.0f}m", flush=True)
                # Upload checkpoint to S3
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", checkpoint_path,
                         f"{S3_WORK}/results_{SHARD_ID}.checkpoint.jsonl", "--quiet"],
                        capture_output=True, timeout=30
                    )
                except Exception:
                    pass

    # Write final output
    with open(output_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    elapsed = time.time() - t0
    print(f"  Done: {len(records)} records in {elapsed/60:.1f}m", flush=True)

    # Upload final results
    subprocess.run(
        ["aws", "s3", "cp", output_path, f"{S3_WORK}/results_{SHARD_ID}.jsonl", "--quiet"],
        capture_output=True
    )
    # Signal done
    done_file = "/tmp/done"
    with open(done_file, "w") as f:
        f.write("done")
    subprocess.run(
        ["aws", "s3", "cp", done_file, f"{S3_WORK}/done_{SHARD_ID}", "--quiet"],
        capture_output=True
    )

if __name__ == "__main__":
    main()
'''

    script = f"""#!/bin/bash
set -ex
export DEBIAN_FRONTEND=noninteractive
export HOME=/root

# Install deps
apt-get update -qq && apt-get install -y -qq unzip curl stockfish python3 python3-pip
pip3 install chess --break-system-packages

# Install AWS CLI
AWSCLI_ARCH=$(uname -m)
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-$AWSCLI_ARCH.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip -q awscliv2.zip && ./aws/install

# Download artifacts
S3_PREFIX="{s3_work}"
mkdir -p /opt/elocator
aws s3 cp $S3_PREFIX/filtered.pgn /opt/elocator/filtered.pgn
aws s3 cp $S3_PREFIX/game_index.json /opt/elocator/game_index.json

# Write worker script
cat << 'WORKER_EOF' > /opt/elocator/worker.py
{worker_py}
WORKER_EOF

# Set environment
export SHARD_ID={shard_id}
export SHARD_START={shard_start}
export SHARD_END={shard_end}
export DEPTH={depth}
export TIME_LIMIT={time_limit}
export SEED={seed}
export THREADS_PER_POS={threads_per_position}
export S3_WORK="{s3_work}"

# Run worker
cd /opt/elocator
python3 worker.py

# Self-terminate
shutdown -h now
"""
    return script


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Sharded Stockfish eval on EC2")
    parser.add_argument("--pgn", required=True, help="Path to filtered.pgn")
    parser.add_argument("--index", required=True, help="Path to game_index.json")
    parser.add_argument("--output", required=True, help="Output merged JSONL path")
    parser.add_argument("--num-shards", type=int, default=1, help="Number of EC2 workers")
    parser.add_argument("--s3-prefix", required=True, help="S3 prefix for artifacts")
    parser.add_argument("--depth", type=int, default=20, help="Stockfish depth (default: 20)")
    parser.add_argument("--time", type=float, default=10.0, help="Time cap per eval in seconds (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--threads-per-position", type=int, default=8,
                        help="Stockfish threads per position (default: 8)")
    parser.add_argument("--instance-type", default="c8g.48xlarge",
                        help="EC2 instance type (default: c8g.48xlarge)")
    parser.add_argument("--poll-interval", type=int, default=60, help="Poll interval seconds")
    parser.add_argument("--cleanup-only", action="store_true", help="Just terminate stale workers")
    args = parser.parse_args(argv)

    config = FleetConfig(
        instance_type=args.instance_type,
        tag_prefix="elo-eval",
        iam_profile="bg-datagen-ec2",
    )

    if args.cleanup_only:
        n = cleanup_stale_workers(config.tag_prefix, config.regions)
        log(f"Cleanup: terminated {n} stale worker(s)")
        return 0

    s3_work = args.s3_prefix.rstrip("/")

    # Load game index to get total count
    with open(args.index) as f:
        offsets = json.load(f)
    total_games = len(offsets)
    log(f"Total games: {total_games:,}")

    # Build shard ranges
    shard_ranges = build_shard_ranges(total_games, args.num_shards)
    for i, (start, end) in enumerate(shard_ranges):
        log(f"  Shard {i}: games {start:,}-{end:,} ({end-start:,} games)")

    # Upload artifacts to S3
    log("Uploading artifacts to S3...")
    pgn_size = os.path.getsize(args.pgn) / 1e9
    log(f"  filtered.pgn ({pgn_size:.1f} GB)")
    s3_cp(args.pgn, f"{s3_work}/filtered.pgn", quiet=False)
    idx_size = os.path.getsize(args.index) / 1e6
    log(f"  game_index.json ({idx_size:.1f} MB)")
    s3_cp(args.index, f"{s3_work}/game_index.json")
    log("  Artifacts uploaded")

    # Clean old done markers
    for i in range(args.num_shards):
        subprocess.run(
            ["aws", "s3", "rm", f"{s3_work}/done_{i}", "--quiet"],
            capture_output=True,
        )

    # Launch fleet
    log(f"Launching {args.num_shards} worker(s)...")
    cpus_per_instance = 192  # c8g.48xlarge

    def make_user_data(worker_id):
        start, end = shard_ranges[worker_id]
        return build_worker_script(
            s3_work=s3_work,
            shard_id=worker_id,
            shard_start=start,
            shard_end=end,
            depth=args.depth,
            time_limit=args.time,
            seed=args.seed,
            threads_per_position=args.threads_per_position,
        )

    workers = launch_fleet(args.num_shards, make_user_data, config)
    launched = sum(1 for w in workers if w.state != WorkerState.FAILED)
    if launched == 0:
        raise RuntimeError("All worker launches failed")

    # Monitor
    log("Monitoring progress...")
    t0 = time.time()
    shard_done = [False] * args.num_shards

    try:
        while True:
            time.sleep(args.poll_interval)
            elapsed = time.time() - t0

            for shard_id in range(args.num_shards):
                if shard_done[shard_id]:
                    continue
                if s3_exists(f"{s3_work}/done_{shard_id}"):
                    shard_done[shard_id] = True
                    log(f"  Shard {shard_id}: DONE ({elapsed:.0f}s)")
                    continue
                ckpt_size = s3_head(f"{s3_work}/results_{shard_id}.checkpoint.jsonl")
                if ckpt_size is not None:
                    log(f"  Shard {shard_id}: checkpoint {ckpt_size/1e6:.1f}MB")
                else:
                    log(f"  Shard {shard_id}: in progress")

            # Health check + relaunch
            check_worker_health(workers)
            occupied = [w.region for w in workers if w.is_active()]
            for w in workers:
                sid = w.worker_id
                if shard_done[sid]:
                    w.state = WorkerState.COMPLETED
                    continue
                if w.state in (WorkerState.RECLAIMED, WorkerState.FAILED):
                    if s3_exists(f"{s3_work}/done_{sid}"):
                        shard_done[sid] = True
                        w.state = WorkerState.COMPLETED
                    else:
                        log(f"  Shard {sid}: worker {w.state.value}, relaunching...")
                        exclude = list(occupied)
                        if w.region and w.region not in exclude:
                            exclude.append(w.region)
                        relaunch_worker(w, make_user_data, config, exclude_regions=exclude)

            done_count = sum(shard_done)
            log(f"  Progress: {done_count}/{args.num_shards} shards ({elapsed/60:.0f}m)")

            if all(shard_done):
                break

        # Merge results
        log("Downloading and merging results...")
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        total_records = 0

        with open(args.output, "w") as out_f:
            for shard_id in range(args.num_shards):
                local_path = os.path.join(tempfile.gettempdir(), f"results_{shard_id}.jsonl")
                s3_download(f"{s3_work}/results_{shard_id}.jsonl", local_path)
                count = 0
                with open(local_path) as f:
                    for line in f:
                        if line.strip():
                            out_f.write(line if line.endswith("\n") else line + "\n")
                            count += 1
                total_records += count
                log(f"  Shard {shard_id}: {count:,} records")

        elapsed = time.time() - t0
        log(f"Merged {total_records:,} records to {args.output}")
        log(f"Total: {elapsed/60:.0f}m ({elapsed/3600:.1f}h)")

        # Upload merged to S3
        try:
            s3_cp(args.output, f"{s3_work}/train_merged.jsonl")
        except RuntimeError:
            pass

    finally:
        log("Terminating fleet...")
        terminate_fleet(workers)

    return 0


if __name__ == "__main__":
    sys.exit(main())
