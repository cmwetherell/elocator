#!/usr/bin/env python3
"""Launch a single EC2 spot instance to evaluate chess positions with Stockfish.

Uploads filtered.pgn + game_index.json to S3, launches a 192-core spot instance
that runs multiprocessed Stockfish evals (24 positions in parallel, 8 threads each),
checkpoints to S3 every 1000 records, self-terminates when done.

Usage:
    poetry run python src/elocator/ec2_eval.py \
        --pgn ./data/filtered.pgn \
        --index ./data/game_index.json \
        --s3-prefix s3://elocator-data/eval_d20_t10s \
        --depth 20 --time 10

    # Check progress:
    aws s3 ls s3://elocator-data/eval_d20_t10s/

    # Download results when done:
    aws s3 cp s3://elocator-data/eval_d20_t10s/results.jsonl ./data/eval_d20_t10s/train.jsonl

    # Kill early:
    poetry run python src/elocator/ec2_eval.py --cleanup
"""

import argparse
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Import ec2_fleet from bg/train for spot price discovery + launch
sys.path.insert(0, str(Path.home() / "dev" / "bg" / "train"))
from ec2_fleet import (
    FleetConfig,
    cleanup_stale_workers,
    get_spot_prices,
    launch_instance,
    lookup_ami,
    terminate_instances,
    log,
    CANDIDATE_REGIONS,
)

# x86_64 instance — Stockfish 18 has official Linux x86_64 binaries with AVX2
DEFAULT_INSTANCE_TYPE = "c7a.48xlarge"


def s3_cp(local_path, s3_path, quiet=True):
    cmd = ["aws", "s3", "cp", local_path, s3_path]
    if quiet:
        cmd.append("--quiet")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"S3 upload failed: {result.stderr[:500]}")


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


def build_user_data(s3_prefix, depth, time_limit, seed, threads_per_pos, max_games=0):
    """Build the bash script that runs on the EC2 instance."""

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

DEPTH = int(os.environ.get("DEPTH", "20"))
TIME_LIMIT = float(os.environ.get("TIME_LIMIT", "10"))
SEED = int(os.environ.get("SEED", "42"))
THREADS_PER_POS = int(os.environ.get("THREADS_PER_POS", "8"))
S3_PREFIX = os.environ["S3_PREFIX"]
MAX_GAMES = int(os.environ.get("MAX_GAMES", "0"))  # 0 = all
CHECKPOINT_INTERVAL = 10000
PGN_PATH = "/opt/elocator/filtered.pgn"

def get_win_percent(centipawns):
    return 50 + 50 * (2 / (1 + 10 ** (-centipawns / 400)) - 1)

def calculate_win_percentage_change(cp_before, cp_after):
    wp_before = get_win_percent(cp_before)
    wp_after = get_win_percent(cp_after)
    return max(0, -(wp_after - wp_before))

def eval_one_game(args):
    game_idx, byte_offset, seed, depth, time_limit, threads = args
    try:
        engine = chess.engine.SimpleEngine.popen_uci("/usr/games/stockfish")
        engine.configure({"Threads": threads})

        with open(PGN_PATH, errors="replace") as f:
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
        try:
            engine.quit()
        except:
            pass
        return None


def main():
    print("Loading game index...", flush=True)
    with open("/opt/elocator/game_index.json") as f:
        offsets = json.load(f)

    total = len(offsets)
    print(f"Total games: {total:,}", flush=True)

    # Deterministic shuffle
    rng = random.Random(SEED)
    shuffled_order = list(range(total))
    rng.shuffle(shuffled_order)

    # Check for resume checkpoint
    checkpoint_path = "/opt/elocator/checkpoint.json"
    results_path = "/opt/elocator/results.jsonl"
    start_from = 0
    records = []

    # Try to download existing checkpoint from S3
    try:
        subprocess.run(
            ["aws", "s3", "cp", f"{S3_PREFIX}/checkpoint.json", checkpoint_path, "--quiet"],
            capture_output=True, timeout=30
        )
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path) as f:
                ckpt = json.load(f)
            start_from = ckpt.get("games_processed", 0)
            print(f"Resuming from position {start_from:,}", flush=True)
            # Download existing results
            subprocess.run(
                ["aws", "s3", "cp", f"{S3_PREFIX}/results.jsonl", results_path, "--quiet"],
                capture_output=True, timeout=60
            )
            if os.path.exists(results_path):
                with open(results_path) as f:
                    for line in f:
                        if line.strip():
                            records.append(json.loads(line))
                print(f"Loaded {len(records):,} existing records", flush=True)
    except Exception as e:
        print(f"No checkpoint found, starting fresh: {e}", flush=True)

    if MAX_GAMES > 0:
        end = min(start_from + MAX_GAMES, total)
    else:
        end = total
    my_games = shuffled_order[start_from:end]
    print(f"Games to evaluate: {len(my_games):,} (positions {start_from:,} to {end:,})", flush=True)

    num_cpus = cpu_count()
    num_workers = max(1, num_cpus // THREADS_PER_POS)
    print(f"{num_cpus} CPUs, {num_workers} parallel workers x {THREADS_PER_POS} threads", flush=True)

    work_items = [
        (game_idx, offsets[game_idx], SEED, DEPTH, TIME_LIMIT, THREADS_PER_POS)
        for game_idx in my_games
    ]

    completed = 0
    t0 = time.time()
    print(f"=== EVAL START {time.strftime('%H:%M:%S')} ===", flush=True)

    with Pool(processes=num_workers) as pool:
        for result in pool.imap_unordered(eval_one_game, work_items, chunksize=4):
            if result is not None:
                records.append(result)
            completed += 1

            # Verbose: log first 10 individually, then every 100, then every 1000
            if completed <= 10 or (completed <= 1000 and completed % 100 == 0) or completed % 1000 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 0
                acc_str = f"acc={result['accuracy']:.2f}" if result else "skipped"
                print(f"  [{completed:,}/{len(my_games):,}] {acc_str} | {rate:.1f} pos/s | {elapsed/60:.1f}m", flush=True)

            if completed % CHECKPOINT_INTERVAL == 0:
                # Write results
                with open(results_path, "w") as f:
                    for r in records:
                        f.write(json.dumps(r) + "\n")
                # Write checkpoint
                with open(checkpoint_path, "w") as f:
                    json.dump({"games_processed": start_from + completed}, f)

                elapsed = time.time() - t0
                rate = completed / elapsed
                remaining = len(my_games) - completed
                eta = remaining / rate if rate > 0 else 0
                print(f"[{start_from + completed:,}/{total:,}] "
                      f"{len(records):,} records | {rate:.1f} pos/s | "
                      f"ETA {eta/3600:.1f}h", flush=True)

                # Upload to S3
                try:
                    subprocess.run(
                        ["aws", "s3", "cp", results_path, f"{S3_PREFIX}/results.jsonl", "--quiet"],
                        capture_output=True, timeout=120
                    )
                    subprocess.run(
                        ["aws", "s3", "cp", checkpoint_path, f"{S3_PREFIX}/checkpoint.json", "--quiet"],
                        capture_output=True, timeout=30
                    )
                except Exception:
                    pass

    # Final upload
    with open(results_path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    with open(checkpoint_path, "w") as f:
        json.dump({"games_processed": start_from + completed}, f)

    subprocess.run(["aws", "s3", "cp", results_path, f"{S3_PREFIX}/results.jsonl", "--quiet"],
                   capture_output=True)
    subprocess.run(["aws", "s3", "cp", checkpoint_path, f"{S3_PREFIX}/checkpoint.json", "--quiet"],
                   capture_output=True)

    # Signal done
    with open("/tmp/done", "w") as f:
        f.write("done")
    subprocess.run(["aws", "s3", "cp", "/tmp/done", f"{S3_PREFIX}/done", "--quiet"],
                   capture_output=True)

    elapsed = time.time() - t0
    print(f"DONE: {len(records):,} records in {elapsed/3600:.1f}h", flush=True)

if __name__ == "__main__":
    main()
'''

    script = f"""#!/bin/bash
set -ex
export DEBIAN_FRONTEND=noninteractive
export HOME=/root

echo "=== SETUP START $(date) ==="

# Install deps
apt-get update -qq && apt-get install -y -qq unzip curl python3 python3-pip
pip3 install chess --break-system-packages

# Install Stockfish 18 from official release
echo "Installing Stockfish 18..."
ARCH=$(uname -m)
if [ "$ARCH" = "x86_64" ]; then
    SF_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-x86-64-avx2.tar"
elif [ "$ARCH" = "aarch64" ]; then
    SF_URL="https://github.com/official-stockfish/Stockfish/releases/download/sf_18/stockfish-ubuntu-aarch64.tar"
else
    echo "Unsupported arch: $ARCH" && exit 1
fi
curl -sL "$SF_URL" -o /tmp/stockfish.tar
cd /tmp && tar xf stockfish.tar
cp /tmp/stockfish/stockfish-ubuntu-* /usr/games/stockfish
chmod +x /usr/games/stockfish
echo "Stockfish version:"
/usr/games/stockfish <<< "quit" 2>&1 | head -1

# Install AWS CLI
AWSCLI_ARCH=$(uname -m)
curl -s "https://awscli.amazonaws.com/awscli-exe-linux-$AWSCLI_ARCH.zip" -o /tmp/awscliv2.zip
cd /tmp && unzip -q awscliv2.zip && ./aws/install

# Download artifacts
echo "=== DOWNLOADING DATA $(date) ==="
S3="{s3_prefix}"
mkdir -p /opt/elocator
aws s3 cp $S3/filtered.pgn /opt/elocator/filtered.pgn
aws s3 cp $S3/game_index.json /opt/elocator/game_index.json
echo "=== DATA DOWNLOADED $(date) ==="

# Write worker script
cat << 'WORKER_EOF' > /opt/elocator/worker.py
{worker_py}
WORKER_EOF

# Set environment
export DEPTH={depth}
export TIME_LIMIT={time_limit}
export SEED={seed}
export THREADS_PER_POS={threads_per_pos}
export MAX_GAMES={max_games}
export S3_PREFIX="{s3_prefix}"

# Background log sync every 30s so we can monitor
(
  while true; do
    sleep 30
    if [ -f /opt/elocator/worker.log ]; then
      aws s3 cp /opt/elocator/worker.log $S3/worker.log --quiet 2>/dev/null || true
    fi
  done
) &
LOG_SYNC_PID=$!

echo "=== STARTING WORKER $(date) ==="

# Run
cd /opt/elocator
python3 worker.py 2>&1 | tee /opt/elocator/worker.log

# Kill log sync, upload final log
kill $LOG_SYNC_PID 2>/dev/null || true
aws s3 cp /opt/elocator/worker.log $S3/worker.log --quiet || true

# Self-terminate
shutdown -h now
"""
    return script


def main(argv=None):
    parser = argparse.ArgumentParser(description="Launch EC2 spot instance for Stockfish eval")
    parser.add_argument("--pgn", help="Path to filtered.pgn")
    parser.add_argument("--index", help="Path to game_index.json")
    parser.add_argument("--s3-prefix", help="S3 prefix (e.g. s3://elocator-data/eval_d20_t10s)")
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument("--time", type=float, default=10.0, help="Time cap per eval in seconds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads-per-pos", type=int, default=8)
    parser.add_argument("--max-games", type=int, default=0, help="Max games to eval (0=all)")
    parser.add_argument("--instance-type", default=DEFAULT_INSTANCE_TYPE)
    parser.add_argument("--poll-interval", type=int, default=120, help="Seconds between S3 polls")
    parser.add_argument("--upload-only", action="store_true", help="Just upload artifacts, don't launch")
    parser.add_argument("--cleanup", action="store_true", help="Terminate stale workers and exit")
    args = parser.parse_args(argv)

    tag_prefix = "elo-eval"

    if args.cleanup:
        for region in CANDIDATE_REGIONS:
            cleanup_stale_workers(tag_prefix, [region])
        return 0

    if not args.pgn or not args.index or not args.s3_prefix:
        parser.error("--pgn, --index, and --s3-prefix are required")

    s3_prefix = args.s3_prefix.rstrip("/")

    # Upload artifacts
    log("Uploading artifacts to S3...")
    pgn_gb = os.path.getsize(args.pgn) / 1e9
    log(f"  filtered.pgn ({pgn_gb:.1f} GB)")
    s3_cp(args.pgn, f"{s3_prefix}/filtered.pgn", quiet=False)
    log(f"  game_index.json")
    s3_cp(args.index, f"{s3_prefix}/game_index.json")
    log("  Done")

    if args.upload_only:
        log("Upload complete (--upload-only)")
        return 0

    # Find cheapest region
    log("Checking spot prices...")
    prices = get_spot_prices(args.instance_type, CANDIDATE_REGIONS)
    if not prices:
        log("ERROR: No spot availability")
        return 1

    region = prices[0].region
    price = prices[0].price_per_hour
    log(f"  Cheapest: {region} at ${price:.4f}/hr")

    # Launch
    # Use x86_64 for c7a instances, arm64 for c8g
    arch = "x86_64" if "c7a" in args.instance_type or "c6a" in args.instance_type else "arm64"
    ami_id = lookup_ami(region, arch=arch)
    user_data_raw = build_user_data(
        s3_prefix, args.depth, args.time, args.seed, args.threads_per_pos, args.max_games
    )
    user_data_b64 = base64.b64encode(user_data_raw.encode()).decode()

    log(f"Launching {args.instance_type} in {region}...")
    instance_id = launch_instance(
        region=region,
        ami_id=ami_id,
        instance_type=args.instance_type,
        iam_profile="bg-datagen-ec2",
        user_data_b64=user_data_b64,
        volume_size_gb=30,
        volume_type="gp3",
        tag_name=f"{tag_prefix}-0",
    )
    log(f"  Instance: {instance_id}")
    log(f"  Region: {region}")
    log(f"  Price: ${price:.4f}/hr")
    games_str = f"{args.max_games:,}" if args.max_games > 0 else "all"
    log(f"  Config: D{args.depth}, {args.time}s cap, {args.threads_per_pos} threads/pos, {games_str} games")

    # Monitor
    log(f"\nMonitoring S3 for progress (poll every {args.poll_interval}s)...")
    log(f"  Results: {s3_prefix}/results.jsonl")
    log(f"  To kill: poetry run python src/elocator/ec2_eval.py --cleanup")
    log("")

    try:
        while True:
            time.sleep(args.poll_interval)

            if s3_exists(f"{s3_prefix}/done"):
                log("DONE! Worker completed.")
                break

            ckpt_size = s3_head(f"{s3_prefix}/results.jsonl")
            if ckpt_size is not None:
                log(f"  Results: {ckpt_size/1e6:.1f} MB")
            else:
                log(f"  Waiting for first checkpoint...")

            # Show tail of worker log if available
            try:
                subprocess.run(
                    ["aws", "s3", "cp", f"{s3_prefix}/worker.log", "/tmp/ec2_worker.log", "--quiet"],
                    capture_output=True, timeout=15
                )
                with open("/tmp/ec2_worker.log") as f:
                    lines = f.readlines()
                if lines:
                    last = lines[-1].strip()
                    log(f"  Worker: {last}")
            except Exception:
                pass

    except KeyboardInterrupt:
        log("\nInterrupted. Instance still running — use --cleanup to terminate.")
        return 0

    # Download results
    log(f"Downloading results...")
    output_dir = os.path.dirname(args.pgn)  # same dir as data
    result_path = f"{output_dir}/eval_d{args.depth}_t{args.time}s_results.jsonl"
    subprocess.run(
        ["aws", "s3", "cp", f"{s3_prefix}/results.jsonl", result_path],
        capture_output=False,
    )
    lines = sum(1 for _ in open(result_path))
    log(f"Downloaded {lines:,} records to {result_path}")

    # Cleanup
    terminate_instances([instance_id], region)
    log("Instance terminated.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
