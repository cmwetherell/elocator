# Elocator Model Comparison Report

## CNN (new) vs MLP (old) — Trained on 105,741 GM positions

## 1. Overall Test-Set Metrics

| Metric | Old MLP (12.3M params) | New CNN (1.9M params) | Winner |
|--------|----------------------|---------------------|--------|
| MAE | 1.6637 | 1.7116 | Old |
| RMSE | 3.6407 | 3.7514 | Old |
| Pearson r | 0.2775 | 0.2173 | Old |
| Spearman ρ | 0.0721 | 0.1775 | New |
| Pred range | [0.599, 5.439] | [0.022, 21.533] | — |
| Pred std | 0.6537 | 1.5559 | — |

## 2. Lift Chart (Decile Analysis)

The key question: when we rank positions by predicted complexity, do higher-predicted positions actually have higher actual complexity?

### New CNN Model — Decile Lift

| Decile | Pred Mean | Actual Mean | Count |
|--------|-----------|-------------|-------|
| 1 | 0.2520 | 0.3244 | 1058 |
| 2 | 0.4278 | 0.5056 | 1067 |
| 3 | 0.5544 | 0.7057 | 1048 |
| 4 | 0.6908 | 0.7974 | 1057 |
| 5 | 0.8563 | 0.9224 | 1058 |
| 6 | 1.0489 | 1.3012 | 1057 |
| 7 | 1.2796 | 1.4935 | 1057 |
| 8 | 1.6236 | 1.7220 | 1058 |
| 9 | 2.1932 | 2.1941 | 1057 |
| 10 | 4.9486 | 3.1599 | 1058 |

### Old MLP Model — Decile Lift

| Decile | Pred Mean | Actual Mean | Count |
|--------|-----------|-------------|-------|
| 1 | 0.6518 | 0.4963 | 1059 |
| 2 | 0.7345 | 0.5391 | 1056 |
| 3 | 0.8666 | 0.7853 | 1058 |
| 4 | 1.0264 | 0.9126 | 1057 |
| 5 | 1.1484 | 0.9053 | 1058 |
| 6 | 1.2383 | 1.1282 | 1057 |
| 7 | 1.3302 | 1.2669 | 1057 |
| 8 | 1.4711 | 1.4690 | 1058 |
| 9 | 1.7724 | 1.9584 | 1057 |
| 10 | 2.8657 | 3.6628 | 1058 |

**Lift Ratio (Decile 10 / Decile 1 actual mean):**
- New CNN: 9.74x
- Old MLP: 7.38x

## 3. Double Lift Chart (Head-to-Head)

For each decile of the new model's predictions, what does the old model predict, and vice versa?

### Sorted by New CNN's Deciles

| Decile | Actual Mean | New CNN Pred | Old MLP Pred |
|--------|-------------|-------------|-------------|
| 1 | 0.3244 | 0.2520 | 0.8747 |
| 2 | 0.5056 | 0.4278 | 0.8890 |
| 3 | 0.7057 | 0.5544 | 0.9787 |
| 4 | 0.7974 | 0.6908 | 1.0865 |
| 5 | 0.9224 | 0.8563 | 1.2011 |
| 6 | 1.3012 | 1.0489 | 1.3153 |
| 7 | 1.4935 | 1.2796 | 1.4047 |
| 8 | 1.7220 | 1.6236 | 1.5661 |
| 9 | 2.1941 | 2.1932 | 1.7245 |
| 10 | 3.1599 | 4.9486 | 2.0661 |

### Sorted by Old MLP's Deciles

| Decile | Actual Mean | New CNN Pred | Old MLP Pred |
|--------|-------------|-------------|-------------|
| 1 | 0.4963 | 0.3675 | 0.6518 |
| 2 | 0.5391 | 0.5602 | 0.7345 |
| 3 | 0.7853 | 0.8237 | 0.8666 |
| 4 | 0.9126 | 1.0482 | 1.0264 |
| 5 | 0.9053 | 1.1954 | 1.1484 |
| 6 | 1.1282 | 1.4391 | 1.2383 |
| 7 | 1.2669 | 1.6218 | 1.3302 |
| 8 | 1.4690 | 1.7853 | 1.4711 |
| 9 | 1.9584 | 2.0826 | 1.7724 |
| 10 | 3.6628 | 2.9514 | 2.8657 |

## 4. Performance by Actual Accuracy Range

How well does each model predict within different ranges of actual complexity?

| Accuracy Range | Count | Actual Mean | New Pred Mean | Old Pred Mean | New MAE | Old MAE | Winner |
|----------------|-------|-------------|--------------|--------------|---------|---------|--------|
| 0 (exact) | 4504 (42.6%) | 0.0000 | 1.3456 | 1.2971 | 1.3456 | 1.2971 | Old |
| 0-0.5 | 2267 (21.4%) | 0.2413 | 1.1617 | 1.1952 | 0.9288 | 0.9539 | New |
| 0.5-1 | 1166 (11.0%) | 0.7122 | 1.2079 | 1.1810 | 0.6907 | 0.5044 | Old |
| 1-2 | 1003 (9.5%) | 1.3985 | 1.3676 | 1.2648 | 0.8054 | 0.4817 | Old |
| 2-5 | 981 (9.3%) | 3.1549 | 1.6074 | 1.4314 | 1.9060 | 1.8059 | Old |
| 5-10 | 388 (3.7%) | 6.9652 | 2.0422 | 1.7099 | 5.0962 | 5.2553 | New |
| 10+ | 266 (2.5%) | 19.9314 | 3.1191 | 2.2344 | 16.8179 | 17.6970 | New |

## 5. Discrimination Power

Can the model distinguish easy moves (accuracy < 0.5) from hard moves (accuracy > 5)?

- Easy moves (accuracy < 0.5): n=6771
- Hard moves (accuracy > 5): n=654

| Model | Easy Pred Mean | Hard Pred Mean | Separation Ratio |
|-------|---------------|----------------|-----------------|
| New CNN | 1.2840 | 2.4802 | 1.93x |
| Old MLP | 1.2630 | 1.9232 | 1.52x |

**Concordance** (% of easy/hard pairs correctly ranked):
- New CNN: 76.0%
- Old MLP: 75.1%

## 6. Prediction Distribution

**New CNN:**
- Mean: 1.3875, Std: 1.5559
- Percentiles: 5th=0.2494, 25th=0.5516, 50th=0.9548, 75th=1.6218, 95th=4.0138

**Old MLP:**
- Mean: 1.3106, Std: 0.6537
- Percentiles: 5th=0.6489, 25th=0.8644, 50th=1.1972, 75th=1.4653, 95th=2.6398

**Actual accuracy distribution:**
- Mean: 1.3125, Std: 3.7677
- Percentiles: 5th=0.0000, 25th=0.0000, 50th=0.1831, 75th=0.9948, 95th=6.0739

## 7. Training Details

| | Old MLP | New CNN |
|--|---------|---------|
| Architecture | 7-layer MLP (780→4096→...→1) | 6-block SE-ResNet (12ch→128→1) |
| Parameters | 12.3M | 1.9M |
| Input | 780-dim flattened + mirrored | 18x8x8 spatial tensor |
| Output | Sigmoid [0,1] | Softplus [0,∞) |
| Loss | MSE | Tweedie (p=1.5) |
| Optimizer | Adam (wd=1e-5) | AdamW (wd=1e-2) |
| LR Schedule | ReduceLROnPlateau | Warmup + Cosine |
| Batch Size | 32 | 256 |
| Training Data | ~53K (half of 105K) | 84.6K (80% of 105K) |
| Best Epoch | Unknown | 9 of 29 (early stopped) |
| Grad Clipping | No | Max norm 1.0 |

## 8. Conclusions

### Which model is better?

**Scorecard: Old MLP wins 3/7 metrics, New CNN wins 4/7 metrics.**

### Analysis

1. **Point accuracy (MAE/RMSE):** The old MLP is marginally better (MAE 1.66 vs 1.71). Both models struggle with the extreme right skew — 42% of moves have accuracy 0 (perfect play by GMs), and the long tail extends to ~89 win%. Neither model is great at predicting the *exact* value.

2. **Ranking ability is where the CNN shines.** The Spearman rank correlation (0.178 vs 0.072) is 2.5x better, and the lift ratio (9.74x vs 7.38x) is 32% higher. This means the CNN is much better at *ordering* positions by difficulty, even if its point estimates are slightly worse. For a product that maps predictions to complexity buckets (1-10), ranking is what matters.

3. **The double lift chart confirms the CNN's superior ranking.** When we sort by the CNN's deciles, actual means increase monotonically from 0.32 to 3.16 — a clean staircase. The old MLP's actuals are less separated (0.50 to 3.66) and the middle deciles (4-7) are nearly flat, meaning the MLP can't distinguish "slightly hard" from "moderately hard" positions.

4. **Discrimination (easy vs hard):** CNN concordance 76.0% vs MLP 75.1%. The CNN gives 1.93x separation vs MLP's 1.52x between easy and hard moves.

5. **The CNN's wider prediction range is informative.** The new model spans [0.02, 21.5] vs MLP's cramped [0.6, 5.4]. The MLP's sigmoid caps it at ~5.4, compressing the entire upper tail. The CNN's Softplus lets it assign dramatically higher scores to truly complex positions — this is useful for the API's 10-bucket mapping.

6. **Overfitting is the CNN's main weakness.** It early-stopped at epoch 9 of 200 — the 1.9M parameter model is severely data-starved with only 84K training samples. With more data, the CNN would improve far more than the MLP (the MLP is already near its representational ceiling with its flat input).

### Bottom Line

**The new CNN is the better model for this product** — despite slightly worse MAE, its ranking ability (what actually drives the 1-10 complexity score) is meaningfully superior. The MLP's slight MAE advantage comes from conservative predictions near the mean, which is unhelpful for discrimination.

However, the CNN is currently overfitting and leaving performance on the table. With the recommendations below, it should pull further ahead.

### Recommendations

1. **More data is the #1 priority.** Both models underperform because 105K samples is small for this task. The CNN architecture should improve significantly with 500K+ samples. Process more games from caissabase.pgn (3.4GB available locally).
2. **Reduce CNN capacity** for this data size: try 64 channels and 4 blocks (~500K params) instead of 128/6. This alone could add 20+ useful training epochs.
3. **Increase regularization:** raise Dropout2d from 0.1 to 0.2-0.3 in residual blocks, and consider mixup augmentation on the input tensors.
4. **Consider Huber loss** as an alternative to Tweedie — it's robust to outliers without the numerical fragility (we had to add clamping to prevent NaN).
5. **Recalibrate API percentile ranges** using the new model's prediction distribution on the validation set, since the prediction scale has changed dramatically.
