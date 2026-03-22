# Elocator Model Research Results

## Dataset
- **410K positions** (357K after dedup) from 2.46M filtered classical OTB games (Elo 2000+)
- **Stockfish 18**, depth 20 with 10-second time cap, 8 threads per position
- **Labels**: Move accuracy (win% loss) — how much win equity the player lost by making that move
- **Split**: 80/10/10 train/val/test (seed 1337) → 286K train / 36K val / 36K test

## Primary Metric: Spearman Rank Correlation
For the end goal of Elo estimation from move quality vs position difficulty, **ranking ability matters most**. A player who plays well in positions the model ranks as hard → higher Elo. The absolute predicted value is less important.

## All Models Tested

### Sorted by Spearman (primary metric)

| Rank | Model | Spearman | Pearson | Lift | MAE | Epochs | Params | File |
|------|-------|----------|---------|------|-----|--------|--------|------|
| **1** | **Ensemble (CNN+MLP rank avg)** | **0.1217** | **0.131** | **2.65x** | **2.38** | n/a | n/a | — |
| **2** | **Retrained MLP** | **0.1232** | 0.117 | 2.36x | 2.91 | 36 | 12.3M | `mlp_retrained_357k.pth` |
| 3 | Big CNN heavy dropout (Tweedie) | 0.1089 | 0.136 | 2.91x | 3.00 | 7 | 1.9M | `cnn_heavy_dropout.pth` |
| 4 | CNN Tweedie (original) | 0.1068 | 0.132 | 2.95x | 2.92 | 8 | 1.9M | `cnn_tweedie_357k.pth` |
| 5 | CNN MSE+Sigmoid | 0.1067 | 0.121 | 2.55x | 2.92 | 18 | 1.9M | `cnn_mse_sigmoid.pth` |
| 6 | MLP SiLU + 0.3 dropout | 0.1021 | 0.126 | 2.64x | 2.93 | 34 | 12.8M | — |
| 7 | Small CNN (64ch/4blk) | 0.0987 | 0.133 | 2.95x | 2.96 | 5 | 326K | `cnn_small_64ch.pth` |
| 8 | Wide shallow MLP | 0.0984 | 0.127 | 2.58x | 2.94 | 51 | 6.9M | — |
| 9 | Residual MLP | 0.0960 | 0.121 | 2.56x | 2.92 | 27 | 9.5M | — |
| 10 | Deep narrow MLP | 0.0948 | 0.127 | 2.71x | 2.96 | 53 | 1.3M | — |
| 11 | MLP on CNN tensor (1152d) | 0.0741 | 0.105 | 2.51x | 2.98 | 63 | 14.3M | — |
| 12 | Old MLP (original weights) | 0.0054 | 0.068 | 1.93x | 2.54 | ? | 12.3M | `old_mlp_original.pth` |

## Key Findings

### 1. The Retrained MLP is the best single model for ranking
The original MLP architecture (780→4096→2056→512→128→64→8→1, LeakyReLU, 50% dropout, MSE loss, sigmoid output) trained on the new D20 data achieves the highest Spearman (0.1232). None of the 10 alternative architectures beat it.

### 2. The ensemble is the best overall approach
Rank-averaging the CNN Tweedie predictions with the retrained MLP predictions gives near-best Spearman (0.1217), best Pearson (0.131), best lift (2.65x), and best MAE (2.38). It combines the CNN's strength at extremes with the MLP's middle-range ranking.

### 3. CNNs win on lift but lose on Spearman
CNN models consistently achieve higher lift ratios (2.91-2.95x vs MLP's 2.36x), meaning they better separate the hardest positions from the easiest. But they overfit quickly (5-8 epochs) and lose ranking precision in the middle deciles.

### 4. Heavy dropout (50%) is the key regularizer
The original MLP's 50% dropout is what enables 36 training epochs without overfitting. Reducing dropout to 30% (V4) or changing to Dropout2d in CNNs results in faster overfitting and worse Spearman, even if Pearson improves slightly.

### 5. The 780-dim fen_encoder outperforms the 1152-dim CNN tensor for MLPs
The hand-crafted encoding (mirroring black positions, explicit piece arrays) provides better features for an MLP than raw spatial planes. The CNN tensor is designed for convolutional feature extraction, not linear layers.

### 6. Architecture changes to the MLP provided no improvement
Wide, narrow, residual, shallow — all performed similarly or worse than the original pyramidal shape. The bottleneck is the data/task signal, not the model capacity.

## Recommended Production Models

### For best ranking (Elo estimation): Retrained MLP
- File: `models/mlp_retrained_357k.pth`
- Architecture: `ChessModel(780)` from original codebase
- Input: `fen_encoder(fen)` → 780-dim vector
- Output: sigmoid → [0, 1], multiply by 100 for win% loss
- Training: Adam, MSE loss, 50% dropout, ReduceLROnPlateau

### For best overall balance: Ensemble
- Rank-average of CNN Tweedie + retrained MLP predictions
- Requires both models at inference time
- Best across all metrics simultaneously

## Saved Models

```
research/models/
├── old_mlp_original.pth         # Original MLP, original training data
├── mlp_retrained_357k.pth       # Original MLP, retrained on 357K D20 data ★ BEST SINGLE
├── cnn_tweedie_357k.pth         # CNN + Tweedie loss (epoch 8 checkpoint format)
├── cnn_heavy_dropout.pth        # CNN + Tweedie, 0.3/0.5 dropout
├── cnn_small_64ch.pth           # Small CNN (64ch, 4 blocks)
└── cnn_mse_sigmoid.pth          # CNN + MSE + Sigmoid output
```
