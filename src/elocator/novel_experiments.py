#!/usr/bin/env python3
"""Three novel architecture experiments for position complexity prediction.

Arch 1: Attention-Pooled CNN — learned spatial attention instead of GAP
Arch 2: CNN with Mixup Training — input blending as regularizer + stochastic depth
Arch 3: CNN Feature Distillation — frozen CNN embeddings → MLP with heavy dropout

Usage:
    poetry run python src/elocator/novel_experiments.py --arch 1
    poetry run python src/elocator/novel_experiments.py --arch 2
    poetry run python src/elocator/novel_experiments.py --arch 3
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
from model_cnn import ChessCNNModel, SEBlock, PreActResBlock
from model_build import tweedie_loss
from run_experiment import evaluate, PreEncodedCNNDataset


# =========================================================================
# Architecture 1: Attention-Pooled CNN
# =========================================================================

class AttentionPool(nn.Module):
    """Learned spatial attention pooling — focuses on squares that matter."""
    def __init__(self, channels):
        super().__init__()
        self.attn_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(channels // 4, 1, 1),
        )

    def forward(self, x):
        # x: (B, C, 8, 8)
        attn_weights = self.attn_conv(x)       # (B, 1, 8, 8)
        attn_weights = attn_weights.view(x.shape[0], -1)  # (B, 64)
        attn_weights = F.softmax(attn_weights, dim=1)      # (B, 64)
        attn_weights = attn_weights.view(x.shape[0], 1, 8, 8)  # (B, 1, 8, 8)

        # Weighted spatial average
        pooled = (x * attn_weights).sum(dim=[2, 3])  # (B, C)
        return pooled


class AttentionCNN(nn.Module):
    """CNN with attention pooling instead of GAP."""
    def __init__(self, channels=128, num_blocks=6, block_dropout=0.1,
                 head_dropout=0.3, stochastic_depth=0.3):
        super().__init__()
        self.stem_conv = nn.Conv2d(12, channels, 3, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm2d(channels)

        self.tower = nn.ModuleList([
            PreActResBlock(channels, 4, block_dropout,
                          drop_path=stochastic_depth * (i / max(1, num_blocks - 1)))
            for i in range(num_blocks)
        ])

        self.head_bn = nn.BatchNorm2d(channels)
        self.attn_pool = AttentionPool(channels)  # <-- instead of AdaptiveAvgPool2d

        # Also keep regular GAP and combine both
        self.gap = nn.AdaptiveAvgPool2d(1)

        meta_size = 12
        # 128 (attn) + 128 (gap) + 12 (meta) = 268
        self.head_mlp = nn.Sequential(
            nn.Linear(channels * 2 + meta_size, 256),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(256, 64),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        board = x[:, 0:12]
        side = x[:, 12, 0, 0].unsqueeze(1)
        castling = x[:, 13:17, 0, 0]
        ep = x[:, 17].sum(dim=1)[:, :7]
        metadata = torch.cat([side, castling, ep], dim=1)

        h = F.silu(self.stem_bn(self.stem_conv(board)))
        for block in self.tower:
            h = block(h)
        h = F.silu(self.head_bn(h))

        # Dual pooling: attention + global average
        h_attn = self.attn_pool(h)       # (B, 128) — focused on important squares
        h_gap = self.gap(h).flatten(1)    # (B, 128) — global context
        h = torch.cat([h_attn, h_gap, metadata], dim=1)  # (B, 268)

        return self.head_mlp(h)


# =========================================================================
# Architecture 2: CNN with Mixup Training
# =========================================================================

class MixupCNNDataset(Dataset):
    """Dataset that applies mixup augmentation."""
    def __init__(self, tensors, accuracy, indices, alpha=0.2):
        self.tensors = tensors
        self.accuracy = accuracy
        self.indices = indices
        self.alpha = alpha

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x1 = self.tensors[j]
        y1 = self.accuracy[j]

        # Pick a random partner for mixup
        k = self.indices[torch.randint(len(self.indices), (1,)).item()]
        x2 = self.tensors[k]
        y2 = self.accuracy[k]

        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha) if self.alpha > 0 else 1.0

        # Blend
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2

        return x_mixed, y_mixed


# =========================================================================
# Architecture 3: CNN Feature Distillation
# =========================================================================

class DistilledMLP(nn.Module):
    """MLP head that operates on frozen CNN embeddings."""
    def __init__(self, embed_dim=128, meta_dim=12):
        super().__init__()
        input_dim = embed_dim + meta_dim  # 140
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    def forward(self, x):
        return self.net(x)


def extract_cnn_embeddings(cnn_model, tensors, indices, device, batch_size=256):
    """Extract 128-dim embeddings + 12-dim metadata from CNN."""
    cnn_model.eval()
    embeddings = []
    metadata_list = []

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            idx = indices[i:i+batch_size]
            x = tensors[idx].to(device)

            # Extract metadata
            side = x[:, 12, 0, 0].unsqueeze(1)
            castling = x[:, 13:17, 0, 0]
            ep = x[:, 17].sum(dim=1)[:, :7]
            meta = torch.cat([side, castling, ep], dim=1)

            # Run through CNN tower (same as forward but stop before head)
            board = x[:, 0:12]
            h = F.silu(cnn_model.stem_bn(cnn_model.stem_conv(board)))
            for block in cnn_model.tower:
                h = block(h)
            h = F.silu(cnn_model.head_bn(h))
            h = cnn_model.gap(h).flatten(1)  # (B, 128)

            embeddings.append(h.cpu())
            metadata_list.append(meta.cpu())

    return torch.cat(embeddings), torch.cat(metadata_list)


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, metadata, accuracy, indices=None):
        self.embeddings = embeddings
        self.metadata = metadata
        self.accuracy = accuracy
        self.indices = indices if indices is not None else torch.arange(len(accuracy))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = self.indices[i]
        x = torch.cat([self.embeddings[j], self.metadata[j]])
        return x, self.accuracy[j]


# =========================================================================
# Training
# =========================================================================

def train_model(model, train_loader, val_loader, device, max_epochs, patience,
                loss_fn, optimizer, scheduler, scheduler_type, name):
    best_val, best_ep, no_imp = float('inf'), 0, 0
    save_path = f'experiments/{name}.pth'
    t0 = time.time()

    for epoch in range(max_epochs):
        model.train()
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(features).squeeze(1)
            loss = loss_fn(preds, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for features, labels in val_loader:
                features, labels = features.to(device), labels.to(device)
                preds = model(features).squeeze(1)
                val_loss += loss_fn(preds, labels).item()
        val_loss /= len(val_loader)

        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step(val_loss)

        tag = ''
        if val_loss < best_val:
            best_val, best_ep, no_imp = val_loss, epoch + 1, 0
            torch.save(model.state_dict(), save_path)
            tag = ' *'
        else:
            no_imp += 1

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Ep {epoch+1:3d}/{max_epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"LR: {lr:.2e} | {elapsed/60:.0f}m{tag}", flush=True)

        if patience > 0 and no_imp >= patience:
            print(f"  Early stop. Best: epoch {best_ep}", flush=True)
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model, best_ep, time.time() - t0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--cnn-data", default="data/eval_d20_t10s/features_cnn.pt")
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}", flush=True)

    # Load data
    data = torch.load(args.cnn_data, map_location='cpu', weights_only=False)
    all_tensors = data['tensors']
    all_accuracy = data['accuracy']
    n = all_accuracy.shape[0]

    gen = torch.Generator().manual_seed(1337)
    indices = torch.randperm(n, generator=gen)
    train_size, val_size = int(0.8 * n), int(0.1 * n)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    print(f"Train: {len(train_idx):,} | Val: {len(val_idx):,} | Test: {len(test_idx):,}", flush=True)

    test_actuals = all_accuracy[test_idx].numpy()

    # =====================================================================
    if args.arch == 1:
        name = "novel1_attention_cnn"
        print(f"\n{'='*60}", flush=True)
        print(f"ARCH 1: Attention-Pooled CNN", flush=True)
        print(f"{'='*60}", flush=True)

        model = AttentionCNN(stochastic_depth=0.3).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {params:,}", flush=True)

        train_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, train_idx)
        val_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, val_idx)
        test_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, test_idx)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)
        test_loader = DataLoader(test_ds, batch_size=256)

        loss_fn = lambda p, t: tweedie_loss(p, t, p=1.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

        model, best_ep, wall = train_model(
            model, train_loader, val_loader, device,
            max_epochs=20, patience=0, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler, scheduler_type='cosine',
            name=name)

    # =====================================================================
    elif args.arch == 2:
        name = "novel2_mixup_cnn"
        print(f"\n{'='*60}", flush=True)
        print(f"ARCH 2: CNN with Mixup Training + Stochastic Depth", flush=True)
        print(f"{'='*60}", flush=True)

        model = ChessCNNModel(stochastic_depth=0.3).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Params: {params:,}", flush=True)

        # Mixup dataset for training, normal for val/test
        train_ds = MixupCNNDataset(all_tensors, all_accuracy, train_idx, alpha=0.4)
        val_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, val_idx)
        test_ds = PreEncodedCNNDataset(all_tensors, all_accuracy, test_idx)

        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=256)
        test_loader = DataLoader(test_ds, batch_size=256)

        loss_fn = lambda p, t: tweedie_loss(p, t, p=1.5)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=5)
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-6)
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

        model, best_ep, wall = train_model(
            model, train_loader, val_loader, device,
            max_epochs=30, patience=0, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler, scheduler_type='cosine',
            name=name)

    # =====================================================================
    elif args.arch == 3:
        name = "novel3_distilled_mlp"
        print(f"\n{'='*60}", flush=True)
        print(f"ARCH 3: CNN Feature Distillation → MLP", flush=True)
        print(f"{'='*60}", flush=True)

        # Load pretrained CNN (our best: stochastic depth)
        print("Loading pretrained CNN for embedding extraction...", flush=True)
        teacher = ChessCNNModel(stochastic_depth=0.3).to(device)
        teacher.load_state_dict(torch.load('experiments/exp6_cnn_stochastic_depth.pth', map_location=device))
        teacher.eval()

        # Extract embeddings for all data
        print("Extracting train embeddings...", flush=True)
        train_emb, train_meta = extract_cnn_embeddings(teacher, all_tensors, train_idx, device)
        print("Extracting val embeddings...", flush=True)
        val_emb, val_meta = extract_cnn_embeddings(teacher, all_tensors, val_idx, device)
        print("Extracting test embeddings...", flush=True)
        test_emb, test_meta = extract_cnn_embeddings(teacher, all_tensors, test_idx, device)

        train_acc = all_accuracy[train_idx]
        val_acc = all_accuracy[val_idx]
        test_acc = all_accuracy[test_idx]

        train_ds = EmbeddingDataset(train_emb, train_meta, train_acc)
        val_ds = EmbeddingDataset(val_emb, val_meta, val_acc)
        test_ds = EmbeddingDataset(test_emb, test_meta, test_acc)

        train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=512)
        test_loader = DataLoader(test_ds, batch_size=512)

        model = DistilledMLP(embed_dim=128, meta_dim=12).to(device)
        params = sum(p.numel() for p in model.parameters())
        print(f"Distilled MLP params: {params:,}", flush=True)

        loss_fn = lambda p, t: tweedie_loss(p, t, p=1.5)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        model, best_ep, wall = train_model(
            model, train_loader, val_loader, device,
            max_epochs=100, patience=15, loss_fn=loss_fn,
            optimizer=optimizer, scheduler=scheduler, scheduler_type='plateau',
            name=name)

    # =====================================================================
    # Evaluate
    # =====================================================================
    model.eval()
    preds = []
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            p = model(features).squeeze(1).cpu()
            preds.extend(p.tolist())

    metrics = evaluate(preds, test_actuals)

    print(f"\n{'='*60}", flush=True)
    print(f"RESULTS: {name}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  Best epoch: {best_ep}", flush=True)
    print(f"  Pearson:  {metrics['pearson']:.4f}", flush=True)
    print(f"  Spearman: {metrics['spearman']:.4f}", flush=True)
    print(f"  MAE:      {metrics['mae']:.4f}", flush=True)
    print(f"  Lift:     {metrics['lift']:.2f}x", flush=True)
    print(f"  Wall time: {wall/3600:.1f}h", flush=True)

    results_path = f'experiments/{name}_results.json'
    with open(results_path, 'w') as f:
        json.dump({**metrics, 'best_epoch': best_ep, 'name': name,
                   'wall_hours': wall/3600}, f, indent=2)
    print(f"DONE", flush=True)


if __name__ == "__main__":
    main()
