"""
Analyze Lichess Elo prediction results.
- Train best LightGBM model with full feature set
- SHAP analysis
- Calibration lift charts
- Filter for 2000+ Elo and repeat
- Find most incredible game performance + cheating assessment
"""

import json
import numpy as np
import pickle
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "elo_pipeline"
FEATURES_FILE = DATA_DIR / "lichess_player_features.jsonl"
RAW_GAMES_FILE = DATA_DIR / "lichess_raw_game_data.jsonl"

FEATURE_COLS = [
    "n_moves", "game_length",
    "mean_actual_loss", "std_actual_loss", "median_loss", "p75_loss", "p90_loss", "max_loss",
    "loss_skewness", "loss_kurtosis", "coeff_variation", "iqr_loss",
    "mean_expected_ensemble", "mean_complexity_score", "std_complexity", "max_complexity",
    "mean_ratio", "mean_residual", "std_residual", "complexity_weighted_loss",
    "pct_perfect", "pct_good", "pct_inaccuracy", "pct_mistake", "pct_blunder",
    "pct_minor_inaccuracy", "pct_major_blunder",
    "mean_loss_very_complex", "mean_loss_complex_mid", "mean_loss_moderate", "mean_loss_simple",
    "n_very_complex", "n_complex_mid", "n_simple",
    "pct_perfect_complex", "pct_perfect_simple", "loss_ratio_complex_simple",
    "mean_loss_opening", "mean_loss_middlegame", "mean_loss_endgame",
    "pct_perfect_opening", "pct_perfect_endgame",
    "accuracy_slope", "complexity_slope",
    "mean_loss_behind", "mean_loss_ahead", "mean_loss_equal",
    "pct_moves_behind", "pct_moves_ahead",
    "longest_good_streak", "longest_perfect_streak",
    "good_streak_ratio", "perfect_streak_ratio", "consecutive_errors",
    "perfect_times_complexity", "blunder_in_simple_rate",
    "mean_eval_change", "std_eval_change",
    "random_noise",
]


def load_records(path=FEATURES_FILE, min_elo=None):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if min_elo and r["elo"] < min_elo:
                    continue
                if min_elo and r.get("opponent_elo") and r["opponent_elo"] < min_elo:
                    continue
                records.append(r)
    return records


def records_to_arrays(records):
    rng = np.random.RandomState(123)
    X, y = [], []
    for r in records:
        row = []
        for col in FEATURE_COLS:
            if col == "random_noise":
                row.append(rng.randn())
            else:
                row.append(r.get(col, 0) or 0)
        X.append(row)
        y.append(r["elo"])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_and_analyze(records, label="all", out_prefix="lichess"):
    print(f"\n{'='*70}")
    print(f"  TRAINING: {label} ({len(records):,} records)")
    print(f"{'='*70}")

    X, y = records_to_arrays(records)
    print(f"Features: {X.shape[1]}, Elo range: {y.min():.0f} - {y.max():.0f}")

    # 70/15/15 split
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    train_end = int(0.70 * n)
    val_end = int(0.85 * n)
    X_train, y_train = X[idx[:train_end]], y[idx[:train_end]]
    X_val, y_val = X[idx[train_end:val_end]], y[idx[train_end:val_end]]
    X_test, y_test = X[idx[val_end:]], y[idx[val_end:]]
    print(f"Train: {len(X_train):,}, Val: {len(X_val):,}, Test: {len(X_test):,}")

    # Train LightGBM with tuned params
    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_COLS)
    val_ds = lgb.Dataset(X_val, label=y_val, feature_name=FEATURE_COLS, reference=train_ds)

    params = {
        "objective": "regression",
        "metric": "mae",
        "boosting_type": "gbdt",
        "learning_rate": 0.03,
        "num_leaves": 127,
        "min_child_samples": 50,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 5,
        "lambda_l1": 0.3,
        "lambda_l2": 1.0,
        "max_depth": -1,
        "verbose": -1,
    }

    model = lgb.train(
        params, train_ds, num_boost_round=10000,
        valid_sets=[train_ds, val_ds], valid_names=["train", "val"],
        callbacks=[lgb.log_evaluation(500), lgb.early_stopping(200)],
    )

    print(f"\nBest iteration: {model.best_iteration}")

    # Evaluate
    for name, Xp, yp in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
        pred = model.predict(Xp)
        mae = np.abs(pred - yp).mean()
        rmse = np.sqrt(((pred - yp) ** 2).mean())
        corr = np.corrcoef(pred, yp)[0, 1]
        print(f"  {name:5s}: MAE={mae:.0f}  RMSE={rmse:.0f}  Pearson={corr:.4f}")

    # Save model
    model_path = DATA_DIR / f"elo_model_{out_prefix}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # --- Feature importance with barometer ---
    importance = model.feature_importance(importance_type="gain")
    noise_imp = dict(zip(FEATURE_COLS, importance)).get("random_noise", 0)
    ranked = sorted(zip(FEATURE_COLS, importance), key=lambda x: x[1], reverse=True)
    n_above = sum(1 for name, imp in ranked if imp > noise_imp and name != "random_noise")
    print(f"\nFeatures above random barometer: {n_above}/{len(FEATURE_COLS)-1}")
    print("\nTop 20 features (gain):")
    max_imp = max(importance) if max(importance) > 0 else 1
    for i, (name, imp) in enumerate(ranked[:20]):
        marker = " <<< RANDOM" if name == "random_noise" else ""
        above = " *" if imp > noise_imp and name != "random_noise" else ""
        bar = "#" * int(imp / max_imp * 30)
        print(f"  {name:32s} {imp:12.0f} {bar}{above}{marker}")

    # --- SHAP ---
    print("\nComputing SHAP values...")
    explainer = shap.TreeExplainer(model)
    # Use subset for SHAP if test set is huge
    shap_n = min(len(X_test), 3000)
    shap_X = X_test[:shap_n]
    shap_values = explainer.shap_values(shap_X)

    fig, ax = plt.subplots(figsize=(12, 16))
    shap.summary_plot(shap_values, shap_X, feature_names=FEATURE_COLS,
                      show=False, max_display=30)
    plt.title(f"SHAP Beeswarm — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(DATA_DIR / f"shap_{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(shap_values, shap_X, feature_names=FEATURE_COLS,
                      plot_type="bar", show=False, max_display=30)
    plt.title(f"SHAP Bar — {label}", fontsize=14)
    plt.tight_layout()
    plt.savefig(DATA_DIR / f"shap_bar_{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    # --- Calibration lift chart (100-Elo buckets by predicted) ---
    test_pred = model.predict(X_test)
    bucket_size = 100
    buckets = {}
    for actual, predicted in zip(y_test, test_pred):
        b = int(predicted // bucket_size) * bucket_size
        if b not in buckets:
            buckets[b] = {"actual": [], "predicted": []}
        buckets[b]["actual"].append(actual)
        buckets[b]["predicted"].append(predicted)

    sorted_buckets = sorted(buckets.keys())
    bucket_centers = [b + bucket_size / 2 for b in sorted_buckets]
    actual_means = [np.mean(buckets[b]["actual"]) for b in sorted_buckets]
    predicted_means = [np.mean(buckets[b]["predicted"]) for b in sorted_buckets]
    counts = [len(buckets[b]["actual"]) for b in sorted_buckets]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(bucket_centers, predicted_means, "s-", color="#FF5722", linewidth=2.5,
             markersize=7, label="Predicted Elo (bucket mean)", zorder=3)
    ax1.plot(bucket_centers, actual_means, "o-", color="#2196F3", linewidth=2.5,
             markersize=7, label="Actual Elo (bucket mean)", zorder=3)
    mn, mx = min(bucket_centers), max(bucket_centers)
    ax1.plot([mn, mx], [mn, mx], "--", color="gray", alpha=0.5, linewidth=1,
             label="Perfect calibration")
    ax1.set_xlabel("Predicted Elo (100-point buckets)", fontsize=13)
    ax1.set_ylabel("Elo", fontsize=13)
    ax1.set_title(f"Calibration Lift Chart — {label}", fontsize=14)
    ax1.legend(loc="upper left", fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(bucket_centers, counts, width=bucket_size * 0.7, alpha=0.12, color="gray", zorder=1)
    ax2.set_ylabel("Sample count", fontsize=11, color="gray")
    ax2.tick_params(axis="y", labelcolor="gray")
    plt.tight_layout()
    plt.savefig(DATA_DIR / f"calibration_{out_prefix}.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nCalibration ({label}):")
    print(f"{'Bucket':>12s} {'Count':>6s} {'Predicted':>10s} {'Actual':>8s} {'Error':>7s}")
    for b, ac, pr, ct in zip(sorted_buckets, actual_means, predicted_means, counts):
        print(f"{b:>5d}-{b+99:>5d} {ct:6d} {pr:10.0f} {ac:8.0f} {ac-pr:+7.0f}")

    print(f"\nSaved: shap_{out_prefix}.png, shap_bar_{out_prefix}.png, calibration_{out_prefix}.png")
    return model, X_test, y_test, test_pred, idx[val_end:]


def find_most_incredible_game(records):
    """Find the game where a player most dramatically outperformed their Elo."""
    print(f"\n{'='*70}")
    print(f"  FINDING MOST INCREDIBLE PERFORMANCE")
    print(f"{'='*70}")

    # Score each record by how much better they played than expected for their Elo
    # Low mean_actual_loss + high complexity = incredible
    # Normalize by Elo bucket expectations
    elo_loss_map = {}
    for r in records:
        bucket = int(r["elo"] // 200) * 200
        if bucket not in elo_loss_map:
            elo_loss_map[bucket] = []
        elo_loss_map[bucket].append(r["mean_actual_loss"])

    elo_expected = {b: np.mean(v) for b, v in elo_loss_map.items()}
    elo_std = {b: np.std(v) for b, v in elo_loss_map.items()}

    scored = []
    for r in records:
        bucket = int(r["elo"] // 200) * 200
        if bucket not in elo_expected or elo_std.get(bucket, 0) < 0.01:
            continue
        # Z-score: how many SDs below the mean loss for their Elo bucket
        z = (elo_expected[bucket] - r["mean_actual_loss"]) / elo_std[bucket]
        # Bonus for doing it in complex positions
        complexity_bonus = r.get("mean_complexity_score", 50) / 50
        score = z * complexity_bonus
        scored.append((score, z, r))

    scored.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 10 most incredible single-game performances:")
    print(f"{'Rank':>4s} {'Player':>25s} {'Elo':>5s} {'Loss%':>6s} {'ExpLoss':>8s} {'Z':>6s} {'Cmplx':>5s} {'Moves':>5s} {'Pct✓':>5s} {'Result':>8s}")
    for i, (score, z, r) in enumerate(scored[:10]):
        bucket = int(r["elo"] // 200) * 200
        exp = elo_expected[bucket]
        print(f"{i+1:>4d} {r['player']:>25s} {r['elo']:>5d} {r['mean_actual_loss']:>6.2f} {exp:>8.2f} {z:>6.1f} {r.get('mean_complexity_score',0):>5.0f} {r['n_moves']:>5d} {r['pct_perfect']*100:>5.1f} {r['result']:>8s}")

    # Deep dive on #1
    best_score, best_z, best = scored[0]
    bucket = int(best["elo"] // 200) * 200
    exp_loss = elo_expected[bucket]
    exp_std = elo_std[bucket]

    print(f"\n{'─'*60}")
    print(f"MOST INCREDIBLE PERFORMANCE:")
    print(f"  Player:             {best['player']}")
    print(f"  Elo:                {best['elo']}")
    print(f"  Opponent Elo:       {best.get('opponent_elo', '?')}")
    print(f"  Color:              {best['color']}")
    print(f"  Result:             {best['result']}")
    print(f"  Time Control:       {best.get('time_control', '?')}")
    print(f"  Moves:              {best['n_moves']}")
    print(f"  Mean accuracy loss: {best['mean_actual_loss']:.3f}%")
    print(f"  Expected for Elo:   {exp_loss:.3f}% (std={exp_std:.3f})")
    print(f"  Z-score:            {best_z:.1f} standard deviations below mean")
    print(f"  Perfect moves:      {best['pct_perfect']*100:.1f}%")
    print(f"  Good moves (<1%):   {best['pct_good']*100:.1f}%")
    print(f"  Blunders:           {best['pct_blunder']*100:.1f}%")
    print(f"  Mean complexity:    {best.get('mean_complexity_score', 0):.1f}")
    print(f"  Mean eval change:   {best.get('mean_eval_change', 0):.1f} cp")

    # Cheating assessment
    print(f"\n  CHEATING LIKELIHOOD ASSESSMENT:")
    flags = 0
    total_flags = 0

    # Flag 1: Perfect move rate for their Elo
    pct_perfect = best["pct_perfect"]
    if pct_perfect > 0.80:
        flags += 2
        print(f"  🔴 Perfect move rate {pct_perfect*100:.1f}% is extremely high")
    elif pct_perfect > 0.65:
        flags += 1
        print(f"  🟡 Perfect move rate {pct_perfect*100:.1f}% is notably high")
    else:
        print(f"  🟢 Perfect move rate {pct_perfect*100:.1f}% is plausible")
    total_flags += 2

    # Flag 2: Z-score magnitude
    if best_z > 4:
        flags += 2
        print(f"  🔴 Z-score of {best_z:.1f} is extreme (>4σ below expected loss)")
    elif best_z > 3:
        flags += 1
        print(f"  🟡 Z-score of {best_z:.1f} is very unusual (>3σ)")
    else:
        print(f"  🟢 Z-score of {best_z:.1f} is within range of a great day")
    total_flags += 2

    # Flag 3: Consistency — low std means robotically consistent
    if best.get("std_actual_loss", 99) < 0.3:
        flags += 2
        print(f"  🔴 Accuracy std={best['std_actual_loss']:.2f} — suspiciously consistent (engine-like)")
    elif best.get("std_actual_loss", 99) < 0.6:
        flags += 1
        print(f"  🟡 Accuracy std={best['std_actual_loss']:.2f} — very consistent")
    else:
        print(f"  🟢 Accuracy std={best['std_actual_loss']:.2f} — normal human variance")
    total_flags += 2

    # Flag 4: Performance in complex positions
    complex_loss = best.get("mean_loss_very_complex", 99)
    if complex_loss < 0.2:
        flags += 2
        print(f"  🔴 Loss in very complex positions: {complex_loss:.2f}% — near-engine level")
    elif complex_loss < 0.5:
        flags += 1
        print(f"  🟡 Loss in complex positions: {complex_loss:.2f}% — unusually good")
    else:
        print(f"  🟢 Loss in complex positions: {complex_loss:.2f}% — human-level")
    total_flags += 2

    # Flag 5: Blunder rate
    blunder_rate = best.get("pct_blunder", 0)
    if blunder_rate == 0 and best["n_moves"] > 25:
        flags += 1
        print(f"  🟡 Zero blunders in {best['n_moves']} moves — unusual but possible")
    else:
        print(f"  🟢 Blunder rate {blunder_rate*100:.1f}% — normal")
    total_flags += 1

    # Flag 6: Elo vs performance gap
    # If they're 1200 playing like 2500, that's suspicious
    if best["elo"] < 1500 and best["mean_actual_loss"] < 0.3:
        flags += 2
        print(f"  🔴 {best['elo']} Elo with {best['mean_actual_loss']:.2f}% loss — massive Elo-performance gap")
    elif best["elo"] < 2000 and best["mean_actual_loss"] < 0.2:
        flags += 1
        print(f"  🟡 {best['elo']} Elo with {best['mean_actual_loss']:.2f}% loss — notable gap")
    else:
        print(f"  🟢 Performance consistent with Elo range")
    total_flags += 2

    pct = flags / total_flags * 100
    if pct >= 70:
        verdict = "VERY LIKELY cheating"
    elif pct >= 50:
        verdict = "SUSPICIOUS — warrants investigation"
    elif pct >= 30:
        verdict = "POSSIBLY having a great day, but unusual"
    else:
        verdict = "LIKELY legitimate — just an outstanding game"
    print(f"\n  Suspicion score: {flags}/{total_flags} ({pct:.0f}%)")
    print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    print("Loading all records...")
    all_records = load_records()
    print(f"Total records: {len(all_records):,}")

    elos = [r["elo"] for r in all_records]
    print(f"Elo range: {min(elos)} - {max(elos)}, Mean: {np.mean(elos):.0f}")

    # --- Full dataset ---
    model_all, X_test_all, y_test_all, pred_all, test_idx_all = train_and_analyze(
        all_records, label="All Lichess Elos", out_prefix="lichess_all"
    )

    # --- 2000+ filter ---
    print("\n\nFiltering for both players 2000+ Elo...")
    records_2k = load_records(min_elo=2000)
    print(f"Records with both players 2000+: {len(records_2k):,}")
    if len(records_2k) >= 100:
        train_and_analyze(records_2k, label="Lichess 2000+ Only", out_prefix="lichess_2k")
    else:
        print("Not enough 2000+ records for meaningful training")

    # --- Most incredible game ---
    find_most_incredible_game(all_records)
