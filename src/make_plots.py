"""Generate metrics tables + ROC/PR/CM/comparison/training-curve plots.

Discovers all gcn_*_metrics.json files alongside xgb_baseline_metrics.json
and produces a unified comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from load_data import PROJECT_ROOT


METRIC_ORDER = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def oof_metrics(y_true: np.ndarray, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
    }


def discover_runs(proc: Path) -> dict[str, tuple[Path, str]]:
    """Return dict run_label -> (oof_csv_path, proba_col)."""
    runs: dict[str, tuple[Path, str]] = {}
    xgb = proc / "xgb_baseline_oof_preds.csv"
    if xgb.exists():
        runs["XGBoost"] = (xgb, "xgb_proba")
    for path in sorted(proc.glob("gcn_*_oof_preds.csv")):
        stem = path.stem.replace("_oof_preds", "")
        # Variant name = whatever follows "gcn_"; or "baseline" / "default" if just "gcn"
        suffix = stem[len("gcn_"):] if stem.startswith("gcn_") else stem
        label = f"GCN[{suffix}]" if suffix else "GCN[default]"
        # The proba column was written as f"{prefix}proba" e.g. "gcn_v2_thr05_proba".
        df = pd.read_csv(path, nrows=1)
        proba_col = next((c for c in df.columns if c.endswith("proba")), None)
        if proba_col is None:
            continue
        runs[label] = (path, proba_col)
    return runs


def collect_models(proc: Path) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    runs = discover_runs(proc)
    if not runs:
        raise SystemExit("no model OOF predictions found in " + str(proc))
    common: pd.Index | None = None
    raw: dict[str, pd.DataFrame] = {}
    for label, (path, proba_col) in runs.items():
        df = pd.read_csv(path, index_col=0).dropna(subset=[proba_col])
        raw[label] = df
        common = df.index if common is None else common.intersection(df.index)
    common = common.sort_values()
    y_true = raw[next(iter(raw))].loc[common, "tp53_binary"].astype(int).values
    models = {label: raw[label].loc[common, proba_col].values
              for label, (_, proba_col) in runs.items()}
    print(f"Models found ({len(models)}):", list(models.keys()))
    print(f"Common cell lines for comparison: {len(common)}")
    return y_true, models


def plot_roc(y_true: np.ndarray, models: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = sns.color_palette("tab10", len(models))
    for (name, proba), color in zip(models.items(), palette):
        fpr, tpr, _ = roc_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color, linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — TP53 mutation status (OOF)")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true: np.ndarray, models: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    palette = sns.color_palette("tab10", len(models))
    baseline = float(np.mean(y_true))
    for (name, proba), color in zip(models.items(), palette):
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=color, linewidth=2)
    ax.axhline(baseline, color="gray", linestyle="--", alpha=0.5,
               label=f"Random ({baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall — TP53 mutation status (OOF)")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_confusion(y_true: np.ndarray, proba: np.ndarray, name: str, out_path: Path) -> None:
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["WT", "Mutant"], yticklabels=["WT", "Mutant"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{name} — Confusion matrix (OOF)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_comparison(metrics: pd.DataFrame, out_path: Path) -> None:
    n = len(metrics)
    fig, ax = plt.subplots(figsize=(max(9, 1.3 * n + 5), 5.5))
    palette = sns.color_palette("tab10", n)
    metrics[METRIC_ORDER].T.plot.bar(ax=ax, color=palette, edgecolor="white", width=0.85)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — TP53 mutation status (OOF)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Model", loc="lower right", fontsize=9)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", padding=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(proc: Path, out_path: Path) -> None:
    curve_files = sorted(proc.glob("gcn_*_curves.csv"))
    if not curve_files:
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    palette = sns.color_palette("tab10", len(curve_files))
    for path, color in zip(curve_files, palette):
        df = pd.read_csv(path)
        run = path.stem.replace("_curves", "").replace("gcn_", "") or "default"
        agg = df.groupby("epoch").agg(
            train_loss_mean=("train_loss", "mean"),
            val_auc_mean=("val_auc", "mean"),
            val_auc_std=("val_auc", "std"),
        )
        axes[0].plot(agg.index, agg["train_loss_mean"], color=color, label=run, linewidth=1.5)
        axes[1].plot(agg.index, agg["val_auc_mean"], color=color, label=run, linewidth=1.5)
        axes[1].fill_between(
            agg.index,
            agg["val_auc_mean"] - agg["val_auc_std"].fillna(0),
            agg["val_auc_mean"] + agg["val_auc_std"].fillna(0),
            color=color, alpha=0.15,
        )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss (mean across folds)")
    axes[0].legend(fontsize=9)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val ROC-AUC")
    axes[1].set_title("Validation ROC-AUC (mean ± std)")
    axes[1].legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def graph_stats(proc: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(proc.glob("gene_graph*.npz")):
        npz = np.load(path, allow_pickle=False)
        ei = npz["edge_index"]
        n_dir = ei.shape[1]
        n_undir = n_dir // 2
        n_genes = len(npz["gene_order"])
        mode = str(npz["mode"]) if "mode" in npz.files else "threshold"
        threshold = float(npz["threshold"]) if "threshold" in npz.files else None
        top_k = int(npz["top_k"]) if "top_k" in npz.files else None
        rows.append({
            "graph": path.stem,
            "n_genes": n_genes,
            "edges_undirected": n_undir,
            "avg_degree": round(n_dir / n_genes, 2),
            "mode": mode,
            "threshold": threshold,
            "top_k": top_k,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    args = parser.parse_args()
    proc = args.proc_dir
    out = proc / "plots"
    out.mkdir(parents=True, exist_ok=True)

    y_true, models = collect_models(proc)

    plot_roc(y_true, models, out / "roc_curves.png")
    plot_pr(y_true, models, out / "pr_curves.png")
    for label, proba in models.items():
        safe = label.replace("[", "_").replace("]", "").replace(" ", "_")
        plot_confusion(y_true, proba, label, out / f"confusion_matrix_{safe}.png")

    rows = {label: oof_metrics(y_true, proba) for label, proba in models.items()}
    metrics_df = pd.DataFrame(rows).T[METRIC_ORDER]
    metrics_df.to_csv(proc / "metrics_table.csv", float_format="%.6f")
    plot_comparison(metrics_df, out / "model_comparison.png")
    plot_training_curves(proc, out / "training_curves.png")

    g_stats = graph_stats(proc)
    if not g_stats.empty:
        g_stats.to_csv(proc / "graph_stats.csv", index=False)
        print("\n=== Graph stats ===")
        print(g_stats.to_string(index=False))

    print("\n=== OOF metrics (n =", len(y_true), ") ===")
    print(metrics_df.round(4).to_string())
    print(f"\nPlots saved → {out}")


if __name__ == "__main__":
    main()
