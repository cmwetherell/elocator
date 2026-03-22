# Stockfish Depth Sensitivity Analysis

## Setup

- **Engine**: Stockfish 18, 8 threads
- **Positions**: 100 randomly sampled from 2.46M filtered classical OTB games (Elo 2000+)
- **Depths tested**: 2, 4, 6, 8, 10, 12, 14, 16, 18
- **Ground truth**: Depth 18
- **Metric**: Move accuracy (win% loss before vs after move)

Each position requires 2 Stockfish evaluations (before-move and after-move states).

## Results

| Depth | MAE | RMSE | Pearson r | Spearman ρ | Concordance | Avg Time | Speedup vs D18 |
|-------|-----|------|-----------|-----------|-------------|----------|----------------|
| 2 | 2.93 | 5.39 | 0.036 | 0.258 | 58.5% | 0.004s | 1183x |
| 4 | 1.42 | 2.41 | 0.536 | 0.486 | 66.4% | 0.004s | 1098x |
| 6 | 1.49 | 2.28 | 0.615 | 0.301 | 60.0% | 0.007s | 684x |
| 8 | 1.29 | 2.06 | 0.680 | 0.445 | 64.9% | 0.023s | 208x |
| 10 | 1.02 | 1.96 | 0.708 | 0.643 | 72.8% | 0.079s | 60x |
| 12 | 0.86 | 1.53 | 0.843 | 0.663 | 73.8% | 0.185s | 25x |
| 14 | 0.63 | 1.06 | 0.927 | 0.708 | 74.8% | 0.504s | 9.3x |
| 16 | 0.52 | 0.90 | 0.946 | 0.740 | 76.6% | 1.56s | 3.0x |
| 18 | 0.00 | 0.00 | 1.000 | 1.000 | 92.9% | 4.69s | 1.0x |

## Error by Position Complexity

Harder positions need deeper search. The "Very Hard" bucket (accuracy > 5 win%) benefits most from additional depth.

| Depth | Easy (0-0.5) | Medium (0.5-2) | Hard (2-5) | Very Hard (5+) |
|-------|-------------|----------------|------------|----------------|
| 6 | 1.05 | 1.42 | 2.10 | 3.21 |
| 8 | 0.78 | 0.93 | 2.19 | 3.66 |
| 10 | 0.44 | 1.21 | 1.19 | 4.15 |
| 12 | 0.31 | 0.85 | 1.70 | 2.74 |
| 14 | 0.19 | 0.71 | 1.23 | 2.13 |
| 16 | 0.18 | 0.50 | 1.23 | 1.24 |

## Key Findings

### 1. Three clear tiers of quality

- **Unusable (D2-D6)**: Pearson < 0.62, Spearman < 0.49. Random-ish for ranking. D6 is actually *worse* than D4 on Spearman (0.30 vs 0.49) — likely a quirk at this sample size, but it shows these shallow depths are noisy.
- **Usable (D8-D12)**: Pearson 0.68-0.84, Spearman 0.44-0.66. Reasonable for training labels. The big jump from D8→D10 (Spearman 0.44→0.64, concordance 65%→73%) marks the entry into "good enough" territory.
- **High quality (D14-D18)**: Pearson > 0.93, Spearman > 0.71. D14 achieves 0.93 Pearson with D18 — meaning D14 and D18 agree on ~93% of the variance.

### 2. The D10→D12 jump is the best cost/quality tradeoff

| Transition | Spearman gain | Concordance gain | Time cost multiplier |
|------------|--------------|------------------|---------------------|
| D8→D10 | +0.198 | +7.9 pp | 3.4x |
| D10→D12 | +0.020 | +1.0 pp | 2.3x |
| D12→D14 | +0.045 | +1.0 pp | 2.7x |
| D14→D16 | +0.032 | +1.8 pp | 3.1x |

D10 is where the Spearman correlation crosses 0.64 and concordance breaks 72%. Beyond D10, each depth step yields diminishing gains for 2-3x more compute.

### 3. D14+ is where "Very Hard" positions stabilize

For easy positions, D10 is already excellent (MAE 0.44). But for the hardest 8% of positions (accuracy > 5 win%), D10 has MAE 4.15 while D14 has 2.13 and D16 has 1.24. If accurate labels on complex tactical positions matter to your use case, D14 is worth the extra cost.

## Recommendation

### For the 2.4M game data generation: **Depth 10**

| Depth | Total time (2.4M games) | Quality (Spearman) |
|-------|------------------------|-------------------|
| 10 | ~53 hours (2.2 days) | 0.643 |
| 12 | ~123 hours (5.1 days) | 0.663 |
| 14 | ~336 hours (14 days) | 0.708 |

**Why D10:**
- Spearman 0.64 and 73% concordance is sufficient for ML training labels — the CNN learns statistical patterns across millions of positions, smoothing individual label noise
- At 0.079s per position, 2.4M games finishes in ~2 days on your 8-core machine
- D12 only gains +0.02 Spearman for 2.3x the cost (an extra 3 days)
- The model's bottleneck is data quantity (overfitting at 84K) not label precision

### If you want higher quality and have the time: **Depth 14**

D14 achieves 0.93 Pearson correlation with D18, meaning it captures 86% of the variance. It's the point of diminishing returns — D16 and D18 add little. But at 14 days for the full dataset, you'd probably want a more powerful machine or to cap at ~500K games.

### Practical suggestion

Start the D10 run now. It'll produce 2.4M training positions in ~2 days. If after training the model still overfits or the lift chart is unsatisfying, re-run a subset at D14 and compare. The `sample_eval.py` script makes this trivial with `--depth 10`.
