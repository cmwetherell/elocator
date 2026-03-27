#!/usr/bin/env python3
"""Pre-encode FENs to tensors for fast training.

Produces two .pt files:
  - features_cnn.pt: {tensors: (N,18,8,8), accuracy: (N,)}
  - features_mlp.pt: {vectors: (N,780), accuracy: (N,)}

Usage:
    poetry run python src/elocator/preprocess_tensors.py \
        --data data/eval_d20_t10s/train_dedup_final.jsonl \
        --out-dir data/eval_d20_t10s/ \
        --workers 8
"""

import argparse
import json
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils import fen_to_tensor, fen_encoder


def encode_cnn(fen):
    return fen_to_tensor(fen).numpy()


def encode_mlp(fen):
    return np.array(fen_encoder(fen), dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    workers = args.workers or cpu_count()

    print(f"Loading {args.data}...", flush=True)
    records = []
    with open(args.data) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    n = len(records)
    print(f"  {n:,} records", flush=True)

    fens = [r['fen'] for r in records]
    accs = np.array([r['accuracy'] for r in records], dtype=np.float32)

    # CNN tensors
    print(f"\nEncoding CNN tensors (18x8x8) with {workers} workers...", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        cnn_arrays = pool.map(encode_cnn, fens, chunksize=1000)
    cnn_tensors = torch.from_numpy(np.stack(cnn_arrays))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({n/elapsed:.0f}/s)", flush=True)
    print(f"  Shape: {cnn_tensors.shape}, Size: {cnn_tensors.nelement()*4/1e9:.1f} GB", flush=True)

    cnn_path = os.path.join(args.out_dir, "features_cnn.pt")
    torch.save({"tensors": cnn_tensors, "accuracy": torch.from_numpy(accs)}, cnn_path)
    print(f"  Saved to {cnn_path}", flush=True)

    # MLP vectors
    print(f"\nEncoding MLP vectors (780-dim) with {workers} workers...", flush=True)
    t0 = time.time()
    with Pool(workers) as pool:
        mlp_arrays = pool.map(encode_mlp, fens, chunksize=1000)
    mlp_vectors = torch.from_numpy(np.stack(mlp_arrays))
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.0f}s ({n/elapsed:.0f}/s)", flush=True)
    print(f"  Shape: {mlp_vectors.shape}, Size: {mlp_vectors.nelement()*4/1e9:.1f} GB", flush=True)

    mlp_path = os.path.join(args.out_dir, "features_mlp.pt")
    torch.save({"vectors": mlp_vectors, "accuracy": torch.from_numpy(accs)}, mlp_path)
    print(f"  Saved to {mlp_path}", flush=True)

    print(f"\nDONE", flush=True)


if __name__ == "__main__":
    main()
