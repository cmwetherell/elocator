"""
Generate all figures and tables for the Elocator paper.

Usage:
    python generate_figures.py                  # generate all figures
    python generate_figures.py --figure 3       # generate only figure 3
    python generate_figures.py --skip-slow       # skip figures requiring 250K game analysis

Requirements:
    - data/game_scores/scored_games.jsonl (OTB scored games)
    - data/elo_pipeline/lichess_player_features.jsonl (Lichess Elo data)
    - data/elo_pipeline/lichess_raw_game_data.jsonl (Lichess raw per-move data)
    - data/elo_pipeline/elo_model_lichess_clean.pkl (trained LightGBM)
    - data/eval_d20_t10s/train_dedup_final.jsonl (OTB test set for lift chart)
    - src/elocator/api/model/ (ensemble model weights + calibration)

Output:
    All figures saved to paper/figures/ as PDF.
"""

import sys
import json
import argparse
import bisect
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import chess
import chess.svg
import cairosvg
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightgbm as lgb
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Paths
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "elocator"
DATA = ROOT / "data"
FIGURES = ROOT / "paper" / "figures"
MODEL_DIR = SRC / "api" / "model"

sys.path.insert(0, str(SRC))
from utils import fen_to_tensor, fen_encoder
from model_cnn import ChessCNNModel, AttentionCNN


# ---------------------------------------------------------------------------
# MLP architecture (needed for ensemble)
# ---------------------------------------------------------------------------
class ChessModel(nn.Module):
    def __init__(self, fen_size):
        super().__init__()
        self.fc1 = nn.Linear(fen_size, 4096)
        self.fc2 = nn.Linear(4096, 2056)
        self.fc3 = nn.Linear(2056, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 8)
        self.fc7 = nn.Linear(8, 1)
        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2056)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn2(self.fc2(x)), negative_slope=0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn3(self.fc3(x)), negative_slope=0.01); x = self.dropout(x)
        x = F.leaky_relu(self.bn4(self.fc4(x)), negative_slope=0.01); x = self.dropout(x)
        x = F.leaky_relu(self.fc5(x), negative_slope=0.01)
        x = F.leaky_relu(self.fc6(x), negative_slope=0.01)
        return torch.sigmoid(self.fc7(x))


def _minmax_norm(val, vmin, vmax):
    return (val - vmin) / (vmax - vmin) if vmax > vmin else 0.5


def load_ensemble():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cnn_sd = ChessCNNModel(stochastic_depth=0.3)
    cnn_sd.load_state_dict(torch.load(MODEL_DIR / "cnn_stochastic_depth.pth", map_location=device))
    cnn_sd.to(device).eval()
    attn_cnn = AttentionCNN(stochastic_depth=0.3)
    attn_cnn.load_state_dict(torch.load(MODEL_DIR / "attention_cnn.pth", map_location=device))
    attn_cnn.to(device).eval()
    mlp = ChessModel(780)
    mlp.load_state_dict(torch.load(MODEL_DIR / "mlp_retrained.pth", map_location=device))
    mlp.to(device).eval()
    with open(MODEL_DIR / "complexity_calibration.json") as f:
        cal = json.load(f)
    return cnn_sd, attn_cnn, mlp, cal, device


def load_scored_games():
    games = []
    with open(DATA / "game_scores" / "scored_games.jsonl") as f:
        for line in f:
            games.append(json.loads(line))
    return games


# ===================================================================
# Figure 1: Decile lift chart (OTB test set)
# ===================================================================
def figure_1_decile_lift():
    print("Figure 1: Decile lift chart...")
    cnn_sd, attn_cnn, mlp, cal, device = load_ensemble()

    records = []
    with open(DATA / "eval_d20_t10s" / "train_dedup_final.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    n = len(records)
    test_records = records[int(0.9 * n):]
    print(f"  Test set: {len(test_records)} positions")

    ensembles, actuals = [], []
    batch_size = 512
    for i in range(0, len(test_records), batch_size):
        chunk = test_records[i:i + batch_size]
        fens = [r["fen"] for r in chunk]
        cnn_t = torch.stack([fen_to_tensor(f) for f in fens]).to(device)
        mlp_t = torch.tensor([fen_encoder(f) for f in fens], dtype=torch.float32).to(device)
        with torch.no_grad():
            sd = cnn_sd(cnn_t).squeeze(1).cpu().numpy()
            at = attn_cnn(cnn_t).squeeze(1).cpu().numpy()
            ml = mlp(mlp_t).squeeze(1).cpu().numpy() * 100
        for j in range(len(chunk)):
            sn = _minmax_norm(float(sd[j]), cal["sd_min"], cal["sd_max"])
            an = _minmax_norm(float(at[j]), cal["attn_min"], cal["attn_max"])
            mn = _minmax_norm(float(ml[j]), cal["mlp_min"], cal["mlp_max"])
            ensembles.append((sn + an + mn) / 3)
            actuals.append(chunk[j]["accuracy"])
        if (i // batch_size) % 50 == 0:
            print(f"    {i}/{len(test_records)}...")

    ensembles = np.array(ensembles)
    actuals = np.array(actuals)
    sorted_idx = np.argsort(ensembles)
    decile_size = len(sorted_idx) // 10
    decile_means = []
    for i in range(10):
        start = i * decile_size
        end = start + decile_size if i < 9 else len(sorted_idx)
        decile_means.append(actuals[sorted_idx[start:end]].mean())

    lift = decile_means[-1] / decile_means[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(1, 11), decile_means, color="#2196F3", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Decile of Predicted Complexity (1=simplest, 10=most complex)", fontsize=12)
    ax.set_ylabel("Actual Mean Win% Loss", fontsize=12)
    ax.set_title(f"Complexity Model Lift Chart — OTB Test Set ({len(test_records):,} positions)\n"
                 f"Actual Win% Loss by Predicted Complexity Decile (Lift = {lift:.1f}x)", fontsize=13)
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels([f"D{i}" for i in range(1, 11)])
    for bar, val in zip(bars, decile_means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "decile_lift.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved. Lift = {lift:.2f}x, D1={decile_means[0]:.2f}, D10={decile_means[-1]:.2f}")


# ===================================================================
# Figure 2: Complexity by move number
# ===================================================================
def figure_2_complexity_by_move():
    print("Figure 2: Complexity by move number...")
    games = load_scored_games()
    move_complexity = defaultdict(list)
    for g in games:
        for i, p in enumerate(g["positions"]):
            move_num = i // 2 + 1
            if move_num <= 60:
                move_complexity[move_num].append(p["complexity"])

    move_nums = sorted(move_complexity.keys())
    means = [np.mean(move_complexity[m]) for m in move_nums]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(move_nums, means, color="#2196F3", linewidth=2)
    ax.fill_between(move_nums, means, alpha=0.15, color="#2196F3")
    ax.set_xlabel("Move Number", fontsize=12)
    ax.set_ylabel("Mean Complexity Score", fontsize=12)
    ax.set_title(f"Position Complexity by Move Number ({len(games):,} OTB games, Elo 2000+)", fontsize=13)
    ax.axvspan(0, 15, alpha=0.06, color="green", label="Opening")
    ax.axvspan(15, 30, alpha=0.06, color="red", label="Middlegame")
    ax.axvspan(30, 60, alpha=0.06, color="blue", label="Endgame")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 60)
    plt.tight_layout()
    plt.savefig(FIGURES / "complexity_by_move.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    opening = [c for m in range(1, 16) for c in move_complexity.get(m, [])]
    middle = [c for m in range(16, 31) for c in move_complexity.get(m, [])]
    endgame = [c for m in range(31, 61) for c in move_complexity.get(m, [])]
    print(f"  Opening: {np.mean(opening):.1f}, Middle: {np.mean(middle):.1f}, Endgame: {np.mean(endgame):.1f}")


# ===================================================================
# Figure 3: Chess features (6-panel)
# ===================================================================
def figure_3_chess_features():
    print("Figure 3: Chess features (slow — analyzes all positions)...")
    games = load_scored_games()
    data = {k: defaultdict(list) for k in ["legal", "captures", "material", "check", "open", "hanging"]}

    total = 0
    for g in games:
        for p in g["positions"]:
            c = p["complexity"]
            board = chess.Board(p["fen"])
            data["legal"][board.legal_moves.count()].append(c)
            data["captures"][min(sum(1 for m in board.legal_moves if board.is_capture(m)), 6)].append(c)
            mat = sum({chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                       chess.QUEEN: 9, chess.KING: 0}[board.piece_at(sq).piece_type]
                      for sq in chess.SQUARES if board.piece_at(sq))
            data["material"][mat].append(c)
            data["check"][board.is_check()].append(c)
            of = sum(1 for f in range(8) if not any(
                board.piece_at(chess.square(f, r)) and board.piece_at(chess.square(f, r)).piece_type == chess.PAWN
                for r in range(8)))
            data["open"][of].append(c)
            h = sum(1 for sq in chess.SQUARES if (p2 := board.piece_at(sq)) and p2.color == board.turn
                    and p2.piece_type != chess.KING and board.is_attacked_by(not board.turn, sq)
                    and not board.is_attacked_by(board.turn, sq))
            data["hanging"][min(h, 3)].append(c)
            total += 1

    print(f"  Analyzed {total:,} positions")
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Legal moves
    ax = axes[0, 0]
    xs = sorted([k for k in data["legal"] if len(data["legal"][k]) > 100 and k <= 50])
    ax.plot(xs, [np.mean(data["legal"][k]) for k in xs], "-", color="#2196F3", linewidth=1.5)
    ax.set_xlabel("Number of Legal Moves"); ax.set_ylabel("Mean Complexity")
    ax.set_title("Legal Moves Available"); ax.grid(True, alpha=0.3)

    # Captures
    ax = axes[0, 1]
    xs = sorted(data["captures"].keys())
    ys = [np.mean(data["captures"][k]) for k in xs]
    labels = [str(x) if x < 6 else "6+" for x in xs]
    bars = ax.bar(labels, ys, color="#FF5722", alpha=0.85)
    ax.set_xlabel("Captures Available"); ax.set_ylabel("Mean Complexity"); ax.set_title("Captures Available")
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Material
    ax = axes[0, 2]
    xs = sorted([k for k in data["material"] if len(data["material"][k]) > 100])
    ax.plot(xs, [np.mean(data["material"][k]) for k in xs], "-", color="#4CAF50", linewidth=1.5)
    ax.set_xlabel("Total Material (pawn units)"); ax.set_ylabel("Mean Complexity")
    ax.set_title("Total Material on Board"); ax.grid(True, alpha=0.3)

    # Check
    ax = axes[1, 0]
    vals = [np.mean(data["check"][False]), np.mean(data["check"][True])]
    ax.bar(["Not in Check", "In Check"], vals, color=["#4CAF50", "#e53935"], alpha=0.85)
    ax.set_ylabel("Mean Complexity"); ax.set_title("Check Status")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    # Open files
    ax = axes[1, 1]
    xs = sorted([k for k in data["open"] if len(data["open"][k]) > 100])
    ys = [np.mean(data["open"][k]) for k in xs]
    bars = ax.bar([str(x) for x in xs], ys, color="#9C27B0", alpha=0.85)
    ax.set_xlabel("Open Files"); ax.set_ylabel("Mean Complexity"); ax.set_title("Open Files")
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # Hanging
    ax = axes[1, 2]
    xs = sorted(data["hanging"].keys())
    ys = [np.mean(data["hanging"][k]) for k in xs]
    labels = [str(x) if x < 3 else "3+" for x in xs]
    bars = ax.bar(labels, ys, color="#FF9800", alpha=0.85)
    ax.set_xlabel("Hanging Pieces"); ax.set_ylabel("Mean Complexity")
    ax.set_title("Hanging (Undefended Attacked) Pieces")
    for bar, v in zip(bars, ys):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle(f"What Makes a Position Complex? ({total:,} positions, {len(games):,} OTB games, Elo 2000+)",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES / "chess_features.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print(f"  Saved.")


# ===================================================================
# Figure 4: Queen x material hierarchical
# ===================================================================
def figure_4_queen_material():
    print("Figure 4: Queen x material hierarchical...")
    games = load_scored_games()
    queen_mat = defaultdict(lambda: defaultdict(list))
    for g in games:
        for p in g["positions"]:
            board = chess.Board(p["fen"])
            c = p["complexity"]
            w_q = len(board.pieces(chess.QUEEN, chess.WHITE))
            b_q = len(board.pieces(chess.QUEEN, chess.BLACK))
            mat = sum({chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
                       chess.QUEEN: 9, chess.KING: 0}[board.piece_at(sq).piece_type]
                      for sq in chess.SQUARES if board.piece_at(sq))
            if w_q > 0 and b_q > 0:
                queen_mat["both"][mat].append(c)
            elif w_q == 0 and b_q == 0:
                queen_mat["none"][mat].append(c)

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, color, marker in [("both", "#e53935", "o"), ("none", "#4CAF50", "s")]:
        mats = sorted([k for k in queen_mat[label] if len(queen_mat[label][k]) > 150])
        means = [np.mean(queen_mat[label][k]) for k in mats]
        nice = "Both queens on" if label == "both" else "Both queens off"
        ax.plot(mats, means, f"{marker}-", color=color, linewidth=2, markersize=5, label=nice)
    ax.set_xlabel("Total Material (pawn units)", fontsize=12)
    ax.set_ylabel("Mean Complexity Score", fontsize=12)
    ax.set_title("Complexity by Material, Stratified by Queen Presence", fontsize=13)
    ax.legend(fontsize=12); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES / "chess_features_hierarchical.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("  Saved.")


# ===================================================================
# Figure 5: Board diagram examples
# ===================================================================
def figure_5_board_examples():
    print("Figure 5: Board diagram examples...")
    positions = [
        ("8/3q2KP/8/2k5/8/8/8/8 w - - 6 61", "very_easy"),
        ("r2qr1k1/3nppbp/1p1p2p1/p7/1n2Q3/2N2N2/PPPB1PPP/1R2R1K1 w - - 0 16", "moderate"),
        ("2kr3r/ppp3pp/3b4/5p2/1R2p3/P1P1P3/1Bq2PPP/4R1K1 w - - 4 20", "hard"),
        ("r3k2r/pb1q2bp/5np1/2p3B1/3p2P1/5P1P/PPPQN1B1/2KR2R1 w kq - 2 18", "very_hard"),
    ]
    for fen, label in positions:
        board = chess.Board(fen)
        svg = chess.svg.board(board, size=400)
        cairosvg.svg2pdf(bytestring=svg.encode(), write_to=str(FIGURES / f"board_{label}.pdf"))
        print(f"  Saved board_{label}.pdf")


# ===================================================================
# Figure 6: Elo calibration chart
# ===================================================================
def figure_6_calibration():
    print("Figure 6: Elo calibration chart...")
    CLEAN_FEATURES, records, X_full, y, test_i, _ = _load_elo_data()
    with open(DATA / "elo_pipeline" / "elo_model_lichess_clean.pkl", "rb") as f:
        model = pickle.load(f)
    pred = model.predict(X_full[test_i])
    y_test = y[test_i]

    buckets = {}
    for actual, predicted in zip(y_test, pred):
        b = int(predicted // 100) * 100
        if b not in buckets:
            buckets[b] = {"actual": [], "predicted": []}
        buckets[b]["actual"].append(actual)
        buckets[b]["predicted"].append(predicted)

    sb = sorted(buckets.keys())
    centers = [b + 50 for b in sb]
    act_m = [np.mean(buckets[b]["actual"]) for b in sb]
    pred_m = [np.mean(buckets[b]["predicted"]) for b in sb]
    counts = [len(buckets[b]["actual"]) for b in sb]

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(centers, pred_m, "s-", color="#FF5722", linewidth=2.5, markersize=7, label="Predicted Elo", zorder=3)
    ax1.plot(centers, act_m, "o-", color="#2196F3", linewidth=2.5, markersize=7, label="Actual Elo", zorder=3)
    ax1.plot([min(centers), max(centers)], [min(centers), max(centers)], "--", color="gray", alpha=0.5, label="Perfect")
    ax1.set_xlabel("Predicted Elo (100-point buckets)", fontsize=13)
    ax1.set_ylabel("Elo", fontsize=13)
    ax1.set_title("Elo Estimation Calibration — 800K Lichess Records", fontsize=14)
    ax1.legend(loc="upper left", fontsize=11); ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=70, alpha=0.12, color="gray", zorder=1)
    ax2.set_ylabel("Sample count", fontsize=11, color="gray"); ax2.tick_params(axis="y", labelcolor="gray")
    plt.tight_layout()
    plt.savefig(FIGURES / "calibration_lichess_all.pdf", format="pdf", bbox_inches="tight")
    plt.close()
    print("  Saved.")


# ===================================================================
# Figure 7: Double lift chart with bootstrap CIs
# ===================================================================
def figure_7_double_lift():
    print("Figure 7: Double lift chart with bootstrap CIs...")
    CLEAN_FEATURES, records, X_full, y, test_i, ACC_ONLY = _load_elo_data()

    with open(DATA / "elo_pipeline" / "elo_model_lichess_clean.pkl", "rb") as f:
        model_a = pickle.load(f)
    pred_a = model_a.predict(X_full[test_i])

    # Train accuracy-only model
    rng2 = np.random.RandomState(123)
    X_acc = []
    for r in records:
        row = []
        for col in ACC_ONLY:
            if col == "random_noise":
                row.append(rng2.randn())
            else:
                row.append(r.get(col, 0) or 0)
        X_acc.append(row)
    X_acc = np.array(X_acc, dtype=np.float32)

    n = len(y)
    idx = np.random.RandomState(42).permutation(n)
    train_end, val_end = int(0.70 * n), int(0.85 * n)
    train_i, val_i = idx[:train_end], idx[train_end:val_end]

    params = {"objective": "regression", "metric": "mae", "boosting_type": "gbdt",
              "learning_rate": 0.03, "num_leaves": 127, "min_child_samples": 50,
              "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 5,
              "lambda_l1": 0.3, "lambda_l2": 1.0, "verbose": -1}
    ds_t = lgb.Dataset(X_acc[train_i], label=y[train_i], feature_name=ACC_ONLY)
    ds_v = lgb.Dataset(X_acc[val_i], label=y[val_i], feature_name=ACC_ONLY, reference=ds_t)
    model_b = lgb.train(params, ds_t, num_boost_round=10000,
                        valid_sets=[ds_t, ds_v], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(0)])
    pred_b = model_b.predict(X_acc[test_i])
    y_test = y[test_i]

    def compute_dl(pa, pb, yt):
        ratio = pa / np.clip(pb, 1, None)
        edges = np.percentile(ratio, np.linspace(0, 100, 21))
        la, lb = [], []
        for i in range(20):
            lo, hi = edges[i], edges[i + 1]
            mask = (ratio >= lo) & (ratio <= hi) if i == 19 else (ratio >= lo) & (ratio < hi)
            if mask.sum() == 0: la.append(1.0); lb.append(1.0); continue
            la.append(yt[mask].mean() / pa[mask].mean())
            lb.append(yt[mask].mean() / pb[mask].mean())
        return np.array(la), np.array(lb)

    boot_a, boot_b = np.zeros((1000, 20)), np.zeros((1000, 20))
    rng_b = np.random.RandomState(0)
    for b in range(1000):
        s = rng_b.choice(len(y_test), len(y_test), replace=True)
        boot_a[b], boot_b[b] = compute_dl(pred_a[s], pred_b[s], y_test[s])

    bcs = list(range(1, 21))
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(bcs, boot_a.mean(0), "o-", color="#2196F3", linewidth=2.5, markersize=6, label="Model A: with complexity")
    ax.fill_between(bcs, np.percentile(boot_a, 2.5, 0), np.percentile(boot_a, 97.5, 0), alpha=0.15, color="#2196F3")
    ax.plot(bcs, boot_b.mean(0), "s-", color="#FF5722", linewidth=2.5, markersize=6, label="Model B: accuracy only")
    ax.fill_between(bcs, np.percentile(boot_b, 2.5, 0), np.percentile(boot_b, 97.5, 0), alpha=0.15, color="#FF5722")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.7, linewidth=1.5, label="Perfect (1.0)")
    ax.set_xlabel("Quantile of pred_A / pred_B ratio", fontsize=11)
    ax.set_ylabel("Actual / Predicted (1.0 = perfect)", fontsize=13)
    ax.set_title("Double Lift Chart with 95% Bootstrap CIs (1,000 resamples)\n"
                 "800K Lichess records, 20 clean features", fontsize=13)
    ax.legend(loc="upper right", fontsize=10); ax.grid(True, alpha=0.3)
    ax.set_ylim(0.85, 1.15); ax.set_xticks(bcs)
    plt.tight_layout()
    plt.savefig(FIGURES / "double_lift_chart.pdf", format="pdf", bbox_inches="tight")
    plt.close()

    overall = (np.abs(boot_a - 1).sum(1) < np.abs(boot_b - 1).sum(1)).mean()
    print(f"  Saved. A better in {overall * 100:.1f}% of bootstraps.")


# ===================================================================
# Figure 8: SHAP beeswarm
# ===================================================================
def figure_8_shap():
    print("Figure 8: SHAP beeswarm...")
    CLEAN_FEATURES, records, X_full, y, test_i, _ = _load_elo_data()
    with open(DATA / "elo_pipeline" / "elo_model_lichess_clean.pkl", "rb") as f:
        model = pickle.load(f)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_full[test_i][:3000])
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_full[test_i][:3000], feature_names=CLEAN_FEATURES, show=False, max_display=15)
    plt.title("SHAP Feature Importance — 800K Lichess Records (top 15)", fontsize=13)
    plt.tight_layout()
    plt.savefig(FIGURES / "shap_lichess_all.pdf", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved.")


# ===================================================================
# Helper: load Elo data
# ===================================================================
CLEAN_FEATURES = [
    "n_moves", "game_length", "mean_expected_ensemble", "std_complexity", "max_complexity",
    "mean_ratio", "mean_residual", "pct_minor_inaccuracy",
    "mean_loss_moderate", "mean_loss_simple", "n_very_complex",
    "mean_loss_opening", "mean_loss_middlegame", "mean_loss_endgame",
    "accuracy_slope", "complexity_slope", "mean_loss_equal",
    "perfect_streak_ratio", "perfect_times_complexity", "std_eval_change",
    "random_noise",
]
COMPLEXITY_FEATURES = {
    "mean_ratio", "mean_residual", "mean_loss_moderate", "mean_loss_simple",
    "n_very_complex", "mean_expected_ensemble", "std_complexity", "max_complexity",
    "complexity_slope", "perfect_times_complexity",
}
ACC_ONLY_FEATURES = [f for f in CLEAN_FEATURES if f not in COMPLEXITY_FEATURES]


def _load_elo_data():
    records = []
    with open(DATA / "elo_pipeline" / "lichess_player_features.jsonl") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                if len(records) >= 800000:
                    break
    rng = np.random.RandomState(123)
    X = []
    for r in records:
        row = []
        for col in CLEAN_FEATURES:
            row.append(rng.randn() if col == "random_noise" else (r.get(col, 0) or 0))
        X.append(row)
    X = np.array(X, dtype=np.float32)
    y = np.array([r["elo"] for r in records], dtype=np.float32)

    n = len(y)
    idx = np.random.RandomState(42).permutation(n)
    test_i = idx[int(0.85 * n):]
    return CLEAN_FEATURES, records, X, y, test_i, ACC_ONLY_FEATURES


# ===================================================================
# Table data: openings, game results
# ===================================================================
def print_table_data():
    print("\nTable data: Opening complexity and game results...")
    games = load_scored_games()
    opening_stats = defaultdict(list)
    for g in games:
        opening_stats[g["headers"].get("Opening", "Unknown")].append(g["summary"]["mean_complexity"])

    ranked = sorted([(k, np.mean(v), len(v)) for k, v in opening_stats.items() if len(v) >= 50],
                    key=lambda x: x[1], reverse=True)
    print("  Most complex openings (min 50 games):")
    for name, mc, n in ranked[:5]:
        print(f"    {name:45s} {mc:.1f} (n={n})")
    print("  Least complex:")
    for name, mc, n in ranked[-5:]:
        print(f"    {name:45s} {mc:.1f} (n={n})")

    decisive = [g["summary"]["mean_complexity"] for g in games if g["headers"].get("Result") in ["1-0", "0-1"]]
    draws = [g["summary"]["mean_complexity"] for g in games if g["headers"].get("Result") == "1/2-1/2"]
    print(f"  Decisive: {np.mean(decisive):.1f} (n={len(decisive):,}), Draws: {np.mean(draws):.1f} (n={len(draws):,})")


# ===================================================================
# Main
# ===================================================================
FIGURE_MAP = {
    1: ("Decile lift chart", figure_1_decile_lift),
    2: ("Complexity by move number", figure_2_complexity_by_move),
    3: ("Chess features (6-panel)", figure_3_chess_features),
    4: ("Queen x material hierarchical", figure_4_queen_material),
    5: ("Board diagram examples", figure_5_board_examples),
    6: ("Elo calibration chart", figure_6_calibration),
    7: ("Double lift chart", figure_7_double_lift),
    8: ("SHAP beeswarm", figure_8_shap),
}

SLOW_FIGURES = {3, 4}  # require iterating all 250K games with chess.Board()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate paper figures")
    parser.add_argument("--figure", type=int, help="Generate only this figure number")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow figures (3, 4)")
    parser.add_argument("--tables", action="store_true", help="Print table data only")
    args = parser.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.tables:
        print_table_data()
    elif args.figure:
        name, func = FIGURE_MAP[args.figure]
        print(f"Generating Figure {args.figure}: {name}")
        func()
    else:
        for num, (name, func) in sorted(FIGURE_MAP.items()):
            if args.skip_slow and num in SLOW_FIGURES:
                print(f"Skipping Figure {num}: {name} (slow)")
                continue
            func()
        print_table_data()
        print("\nAll figures generated.")
