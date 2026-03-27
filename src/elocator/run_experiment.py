#!/usr/bin/env python3
"""Unified experiment runner for complexity model training.

Loads pre-encoded tensors, trains a model, evaluates on fixed holdout.

Usage:
    poetry run python src/elocator/run_experiment.py \
        --model cnn --name "exp2_cnn_baseline" --epochs 15
    poetry run python src/elocator/run_experiment.py \
        --model mlp --name "exp1_mlp_baseline" --epochs 50 --patience 20
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from model_cnn import ChessCNNModel
from model_build import tweedie_loss


# ---------------------------------------------------------------------------
# Standardized evaluation harness
# ---------------------------------------------------------------------------

def evaluate(preds, actuals):
    """Compute all metrics. Returns dict."""
    preds, actuals = np.array(preds), np.array(actuals)
    pearson = np.corrcoef(preds, actuals)[0, 1] if np.std(preds) > 0 else 0.0
    spear, _ = spearmanr(preds, actuals)
    mae = np.abs(preds - actuals).mean()
    df = pd.DataFrame({'p': preds, 'a': actuals})
    df['d'] = pd.qcut(df['p'], 10, labels=False, duplicates='drop') + 1
    lift = df.groupby('d')['a'].mean()
    ratio = lift.iloc[-1] / lift.iloc[0] if lift.iloc[0] > 0 else 0
    return {'pearson': pearson, 'spearman': spear, 'mae': mae, 'lift': ratio}


# ---------------------------------------------------------------------------
# Datasets (pre-encoded)
# ---------------------------------------------------------------------------

class PreEncodedCNNDataset(Dataset):
    def __init__(self, tensors, accuracy, indices, augment=False):
        self.tensors = tensors
        self.accuracy = accuracy
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = self.tensors[j]
        y = self.accuracy[j]

        if self.augment and torch.rand(1).item() < 0.5:
            # Board mirror: flip left-right
            x = x.clone()
            x[:12] = x[:12].flip(-1)          # Flip piece planes
            x[13], x[14] = x[14].clone(), x[13].clone()  # Swap white K/Q castling
            x[15], x[16] = x[16].clone(), x[15].clone()  # Swap black K/Q castling
            x[17] = x[17].flip(-1)             # Mirror EP file

        return x, y


class PreEncodedMLPDataset(Dataset):
    def __init__(self, vectors, accuracy, indices):
        self.vectors = vectors
        self.accuracy = accuracy
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        return self.vectors[j], self.accuracy[j]


# ---------------------------------------------------------------------------
# MLP architecture
# ---------------------------------------------------------------------------

class ChessModel(nn.Module):
    def __init__(self, fen_size=780):
        super().__init__()
        self.fc1 = nn.Linear(fen_size, 4096)
        self.fc2 = nn.Linear(4096, 2056)
        self.fc3 = nn.Linear(2056, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)
        self.fc7 = nn.Linear(8, 1)
        nn.init.kaiming_normal_(self.fc1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc2.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc3.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc4.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc5.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc6.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.fc7.weight)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2056)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), 0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), 0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), 0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), 0.01); x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), 0.01)
        x = F.leaky_relu(self.fc6(x), 0.01)
        return torch.sigmoid(self.fc7(x))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_experiment(args):
    torch.manual_seed(0)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)
    print(f"Experiment: {args.name}", flush=True)

    # Load pre-encoded data
    if args.model == 'cnn':
        data = torch.load(args.cnn_data, map_location='cpu', weights_only=False)
        all_tensors = data['tensors']
        all_accuracy = data['accuracy']
    else:
        data = torch.load(args.mlp_data, map_location='cpu', weights_only=False)
        all_tensors = data['vectors']
        all_accuracy = data['accuracy'] / 100.0  # MLP uses [0,1] scale

    n = all_accuracy.shape[0]
    print(f"Data: {n:,} positions", flush=True)

    # Split (same seed as all previous experiments)
    gen = torch.Generator().manual_seed(1337)
    indices = torch.randperm(n, generator=gen)
    train_size, val_size = int(0.8 * n), int(0.1 * n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}", flush=True)

    # Create datasets
    if args.model == 'cnn':
        train_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, train_idx, augment=args.augment)
        val_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, val_idx)
        test_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, test_idx)
    else:
        train_ds = PreEncodedMLPDataset(all_tensors, all_accuracy, train_idx)
        val_ds = PreEncodedMLPDataset(all_tensors, all_accuracy, val_idx)
        test_ds = PreEncodedMLPDataset(all_tensors, all_accuracy, test_idx)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # Create model
    if args.model == 'cnn':
        model = ChessCNNModel(
            block_dropout=args.block_dropout,
            head_dropout=args.head_dropout,
            stochastic_depth=args.stochastic_depth,
        ).to(device)
    else:
        model = ChessModel().to(device)

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model.upper()}, {params:,} params", flush=True)

    # Loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    elif args.loss == 'huber':
        criterion = nn.SmoothL1Loss(beta=args.huber_beta)
    elif args.loss == 'tweedie':
        criterion = lambda pred, target: tweedie_loss(pred, target, p=1.5)
    print(f"Loss: {args.loss}", flush=True)

    # Optimizer
    if args.model == 'cnn':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs - 5), eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    best_val, best_ep, no_imp = float('inf'), 0, 0
    save_path = os.path.join(args.out_dir, f"{args.name}.pth")
    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()

            if args.model == 'cnn' and args.loss == 'tweedie':
                preds = model(features).squeeze(1)
                loss = criterion(preds, labels)
            elif args.model == 'mlp':
                preds = model(features).squeeze(1)
                if args.log_target:
                    loss = criterion(preds, labels)  # labels already log-transformed
                else:
                    loss = criterion(preds, labels)
            else:
                preds = model(features).squeeze(1)
                loss = criterion(preds, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features).squeeze(1)
                if args.loss == 'tweedie':
                    val_loss += tweedie_loss(preds, labels, p=1.5).item()
                else:
                    val_loss += criterion(preds, labels).item()
        val_loss /= len(val_loader)

        # Scheduler
        if args.model == 'cnn':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        # Best model
        tag = ''
        if val_loss < best_val:
            best_val, best_ep, no_imp = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), save_path)
            tag = ' *'
        else:
            no_imp += 1

        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0
        print(f"  Ep {epoch+1:3d}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | {elapsed/60:.0f}m{tag}", flush=True)

        if args.patience > 0 and no_imp >= args.patience:
            print(f"  Early stop. Best: epoch {best_ep}", flush=True)
            break

    # Load best model
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.eval()

    # Test evaluation
    preds, actuals = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            p = model(features).squeeze(1).cpu()
            if args.model == 'mlp':
                if args.log_target:
                    p = torch.expm1(p) / 100.0  # undo log transform
                p = p * 100  # back to raw accuracy scale
            preds.extend(p.tolist())
            actuals.extend((labels * (100 if args.model == 'mlp' else 1)).tolist())

    metrics = evaluate(preds, actuals)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {args.name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Best epoch: {best_ep}", flush=True)
    print(f"  Pearson:  {metrics['pearson']:.4f}", flush=True)
    print(f"  Spearman: {metrics['spearman']:.4f}", flush=True)
    print(f"  MAE:      {metrics['mae']:.4f}", flush=True)
    print(f"  Lift:     {metrics['lift']:.2f}x", flush=True)
    print(f"  Wall time: {(time.time()-t0)/3600:.1f}h", flush=True)
    print(f"  Saved to: {save_path}", flush=True)

    # Save results
    results_path = os.path.join(args.out_dir, f"{args.name}_results.json")
    with open(results_path, 'w') as f:
        json.dump({**metrics, 'best_epoch': best_ep, 'name': args.name,
                   'model': args.model, 'loss': args.loss, 'epochs': args.epochs,
                   'train_size': len(train_idx), 'wall_hours': (time.time()-t0)/3600}, f, indent=2)
    print(f"DONE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=['cnn', 'mlp'])
    parser.add_argument("--name", required=True, help="Experiment name")
    parser.add_argument("--cnn-data", default="data/eval_d20_t10s/features_cnn.pt")
    parser.add_argument("--mlp-data", default="data/eval_d20_t10s/features_mlp.pt")
    parser.add_argument("--out-dir", default="experiments/")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20, help="0 to disable early stopping")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", default="mse", choices=['mse', 'huber', 'tweedie'])
    parser.add_argument("--huber-beta", type=float, default=1.0)
    parser.add_argument("--log-target", action="store_true", help="Predict log1p(accuracy)")
    parser.add_argument("--block-dropout", type=float, default=0.1, help="CNN block Dropout2d")
    parser.add_argument("--head-dropout", type=float, default=0.3, help="CNN head Dropout")
    parser.add_argument("--augment", action="store_true", help="CNN board mirroring augmentation")
    parser.add_argument("--stochastic-depth", type=float, default=0.0, help="CNN stochastic depth max drop rate")
    args = parser.parse_args()

    if args.model == 'cnn' and args.loss == 'mse':
        args.loss = 'tweedie'  # CNN default

    train_experiment(args)
