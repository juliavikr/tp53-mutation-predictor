"""Threshold calibration analysis for the CCLE-trained XGBoost on TCGA.

Two operating-point options compared against the default 0.5:
  1. F1-optimal threshold derived from CCLE OOF predictions (training-cohort tuned)
  2. Prevalence-matched threshold (chosen so TCGA predicted positive rate matches
     TCGA's actual TP53 mutation rate)

Outputs (data/processed/):
  - threshold_calibration.json          — chosen thresholds + TCGA metrics for each
  - threshold_calibration_table.csv     — same in flat tabular form
  - plots/threshold_calibration.png     — F1 curve over thresholds + chosen points
  - plots/calibration_curve.png         — reliability diagram (CCLE + TCGA, deciles)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from load_data import PROJECT_ROOT


def metrics_at(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "predicted_positive_rate": float(pred.mean()),
    }


def f1_optimal_threshold(y_true: np.ndarray, proba: np.ndarray,
                          grid: np.ndarray | None = None) -> tuple[float, np.ndarray, np.ndarray]:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 181)
    f1s = np.array([
        f1_score(y_true, (proba >= t).astype(int), zero_division=0) for t in grid
    ])
    best_idx = int(np.argmax(f1s))
    return float(grid[best_idx]), grid, f1s


def prevalence_matched_threshold(proba: np.ndarray, target_rate: float) -> float:
    """Pick threshold so that predicted positive rate ≈ target_rate."""
    return float(np.quantile(proba, 1.0 - target_rate))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # CCLE OOF preds on the SHARED 1851-gene feature set (matches what was applied to TCGA)
    ccle = pd.read_csv(proc / "ccle_shared_xgb_oof_preds.csv", index_col=0)
    y_ccle = ccle["tp53_binary"].astype(int).values
    p_ccle = ccle["xgb_shared_proba"].values
    print(f"CCLE OOF: n={len(y_ccle)}  pos_rate={y_ccle.mean():.4f}")

    tcga = pd.read_csv(proc / "tcga_xgb_oof_preds.csv")
    y_tcga = tcga["tp53_binary"].astype(int).values
    p_tcga = tcga["xgb_proba"].values
    print(f"TCGA   : n={len(y_tcga)}  pos_rate={y_tcga.mean():.4f}  "
          f"roc_auc={roc_auc_score(y_tcga, p_tcga):.4f}")

    # ── Choose thresholds ──────────────────────────────────────────────
    t_default = 0.5
    t_f1_ccle, grid, f1s_ccle = f1_optimal_threshold(y_ccle, p_ccle)
    t_prev = prevalence_matched_threshold(p_tcga, y_tcga.mean())
    print(f"\nThresholds:")
    print(f"  default            : 0.500")
    print(f"  F1-optimal on CCLE : {t_f1_ccle:.3f}")
    print(f"  prevalence-matched : {t_prev:.3f}")

    # ── Evaluate on TCGA ───────────────────────────────────────────────
    rows = []
    for label, t in [("default_0.5", t_default),
                     ("f1_optimal_ccle", t_f1_ccle),
                     ("prevalence_matched", t_prev)]:
        m = metrics_at(y_tcga, p_tcga, t)
        m["strategy"] = label
        rows.append(m)
    tab = pd.DataFrame(rows)[
        ["strategy", "threshold", "accuracy", "precision", "recall",
         "f1", "predicted_positive_rate"]
    ]
    print("\nTCGA metrics under each threshold:")
    print(tab.round(4).to_string(index=False))

    # ── Persist ─────────────────────────────────────────────────────────
    summary = {
        "ccle_n": int(len(y_ccle)),
        "tcga_n": int(len(y_tcga)),
        "ccle_positive_rate": float(y_ccle.mean()),
        "tcga_positive_rate": float(y_tcga.mean()),
        "tcga_roc_auc": float(roc_auc_score(y_tcga, p_tcga)),
        "tcga_pr_auc": float(average_precision_score(y_tcga, p_tcga)),
        "thresholds": {
            "default_0.5": 0.5,
            "f1_optimal_ccle": float(t_f1_ccle),
            "prevalence_matched": float(t_prev),
        },
        "tcga_metrics_by_strategy": rows,
    }
    with open(proc / "threshold_calibration.json", "w") as f:
        json.dump(summary, f, indent=2)
    tab.to_csv(proc / "threshold_calibration_table.csv", index=False)

    # ── Plot 1: F1 vs threshold (CCLE + TCGA) ──────────────────────────
    f1s_tcga = np.array([
        f1_score(y_tcga, (p_tcga >= t).astype(int), zero_division=0) for t in grid
    ])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(grid, f1s_ccle, color="steelblue", linewidth=2, label="CCLE OOF F1")
    ax.plot(grid, f1s_tcga, color="tomato", linewidth=2, label="TCGA F1 (CCLE-trained)")
    for t, color, name in [
        (0.5, "gray", "default"),
        (t_f1_ccle, "steelblue", f"F1-opt CCLE = {t_f1_ccle:.2f}"),
        (t_prev, "tomato", f"prev-matched = {t_prev:.2f}"),
    ]:
        ax.axvline(t, color=color, linestyle="--", alpha=0.6)
        ax.text(t + 0.005, 0.05, name, fontsize=8, color=color, rotation=90,
                va="bottom")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("F1 score")
    ax.set_title("Threshold sensitivity — XGBoost on shared 1,851 genes")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="lower center")
    plt.tight_layout()
    plt.savefig(plots / "threshold_calibration.png", dpi=150)
    plt.close()

    # ── Plot 2: reliability diagram (calibration curve) ───────────────
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, y, p, color in [
        (f"CCLE OOF (n={len(y_ccle)})", y_ccle, p_ccle, "steelblue"),
        (f"TCGA (CCLE-trained, n={len(y_tcga)})", y_tcga, p_tcga, "tomato"),
    ]:
        prob_true, prob_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(prob_pred, prob_true, "o-", color=color, linewidth=2,
                markersize=7, label=label)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability (per decile)")
    ax.set_ylabel("Empirical positive rate")
    ax.set_title("Reliability diagram — XGBoost, CCLE vs TCGA")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots / "calibration_curve.png", dpi=150)
    plt.close()

    print(f"\nSaved → {proc} (threshold_calibration.json, _table.csv, plots/threshold_calibration.png, plots/calibration_curve.png)")


if __name__ == "__main__":
    main()
