"""External validation of the CCLE-trained XGBoost on TCGA primary tumours.

Pipeline:
1. Load CCLE expression + labels and TCGA expression + labels.
2. Compute the gene intersection between CCLE top-2k HVG and TCGA features.
3. Within-cohort baseline: 5-fold stratified CV on CCLE restricted to the intersection.
4. Train final XGBoost on ALL CCLE samples (intersection genes) → apply to TCGA.
5. Report Acc / Precision / Recall / F1 / ROC-AUC / PR-AUC for both, plus the gap.
6. Save OOF predictions, plots (ROC/PR/CM for TCGA), and a summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
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
from sklearn.model_selection import StratifiedKFold

from load_data import PROJECT_ROOT, load_ccle
from train_xgb import _xgb_clf


def parse_symbol(g: str) -> str:
    return g.split(" (")[0]


def metrics_from(y_true: np.ndarray, proba: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "positive_rate": float(np.mean(y_true)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    expr_ccle, labels_ccle = load_ccle()
    y_ccle = labels_ccle["tp53_binary"]
    print(f"CCLE: {expr_ccle.shape}  pos_rate={y_ccle.mean():.3f}")

    expr_tcga = pd.read_csv(proc / "tcga_expression.csv", index_col=0)
    labels_tcga = pd.read_csv(proc / "tcga_labels.csv", index_col=0)
    y_tcga = labels_tcga["tp53_binary"].astype(int)
    print(f"TCGA: {expr_tcga.shape}  pos_rate={y_tcga.mean():.3f}")

    top_genes_full = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    symbol_to_full = {parse_symbol(g): g for g in top_genes_full}
    tcga_set = set(expr_tcga.columns)
    shared_symbols = [s for s in symbol_to_full if s in tcga_set]
    print(f"Shared genes (CCLE top-2k ∩ TCGA): {len(shared_symbols)}")

    X_ccle_raw = expr_ccle[[symbol_to_full[s] for s in shared_symbols]].values
    X_tcga_raw = expr_tcga[shared_symbols].values

    # Per-cohort z-score normalisation (essential for cross-cohort transfer:
    # CCLE log2(TPM+1) and TCGA log2(norm_count+1) are on different scales).
    ccle_mu, ccle_sd = X_ccle_raw.mean(axis=0), X_ccle_raw.std(axis=0) + 1e-6
    tcga_mu, tcga_sd = X_tcga_raw.mean(axis=0), X_tcga_raw.std(axis=0) + 1e-6
    X_ccle = (X_ccle_raw - ccle_mu) / ccle_sd
    X_tcga = (X_tcga_raw - tcga_mu) / tcga_sd
    print(f"After per-cohort z-score:  CCLE mean≈{X_ccle.mean():.3f}  TCGA mean≈{X_tcga.mean():.3f}")

    # ── Within-cohort CCLE 5-fold CV on shared genes ──────────────────
    print(f"\nWithin-cohort CCLE {args.n_splits}-fold CV on {len(shared_symbols)} genes...")
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    ccle_oof = np.zeros(len(y_ccle))
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_ccle, y_ccle.values)):
        clf = _xgb_clf(args.seed)
        clf.fit(X_ccle[train_idx], y_ccle.values[train_idx])
        ccle_oof[test_idx] = clf.predict_proba(X_ccle[test_idx])[:, 1]
        print(f"  fold {fold}: trained on {len(train_idx)}, evaluated on {len(test_idx)}")

    ccle_oof_metrics = metrics_from(y_ccle.values, ccle_oof)
    print("\nCCLE OOF (5-fold) on shared genes:")
    for k, v in ccle_oof_metrics.items():
        print(f"  {k:>14s}: {v:.4f}" if isinstance(v, float) else f"  {k:>14s}: {v}")

    # ── Train final on ALL CCLE, predict on TCGA ──────────────────────
    print("\nTraining final XGBoost on full CCLE (no holdout), applying to TCGA...")
    clf_final = _xgb_clf(args.seed)
    clf_final.fit(X_ccle, y_ccle.values)
    tcga_proba = clf_final.predict_proba(X_tcga)[:, 1]

    tcga_metrics = metrics_from(y_tcga.values, tcga_proba)
    print("\nTCGA external metrics (CCLE-trained model):")
    for k, v in tcga_metrics.items():
        print(f"  {k:>14s}: {v:.4f}" if isinstance(v, float) else f"  {k:>14s}: {v}")

    # Per-cancer-type breakdown
    by_type = []
    for ct, sub in labels_tcga.groupby("cancer_type"):
        if len(sub) < 30:
            continue
        idx = labels_tcga.index.get_indexer(sub.index)
        m = metrics_from(y_tcga.values[idx], tcga_proba[idx])
        m["cancer_type"] = ct
        by_type.append(m)
    by_type_df = pd.DataFrame(by_type).sort_values("roc_auc", ascending=False)
    by_type_df.to_csv(proc / "tcga_xgb_per_cancer_type.csv", index=False)
    print(f"\nPer-cancer-type results saved (≥30 samples per type, {len(by_type_df)} types).")

    # Save predictions + summary
    pd.DataFrame({
        "sample_id": labels_tcga.index,
        "tp53_binary": y_tcga.values,
        "cancer_type": labels_tcga["cancer_type"].values,
        "xgb_proba": tcga_proba,
    }).to_csv(proc / "tcga_xgb_oof_preds.csv", index=False)

    pd.DataFrame({
        "ModelID": expr_ccle.index,
        "tp53_binary": y_ccle.values,
        "xgb_shared_proba": ccle_oof,
    }).to_csv(proc / "ccle_shared_xgb_oof_preds.csv", index=False)

    summary = {
        "n_shared_genes": len(shared_symbols),
        "ccle_oof_5fold_on_shared": ccle_oof_metrics,
        "tcga_external": tcga_metrics,
        "drop_ccle_to_tcga": {
            "accuracy": ccle_oof_metrics["accuracy"] - tcga_metrics["accuracy"],
            "f1": ccle_oof_metrics["f1"] - tcga_metrics["f1"],
            "roc_auc": ccle_oof_metrics["roc_auc"] - tcga_metrics["roc_auc"],
            "pr_auc": ccle_oof_metrics["pr_auc"] - tcga_metrics["pr_auc"],
        },
    }
    with open(proc / "tcga_eval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {proc / 'tcga_eval_summary.json'}")

    # ── Plots ─────────────────────────────────────────────────────────
    # ROC: CCLE OOF vs TCGA external
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, y, p, color in [
        (f"CCLE OOF (n={ccle_oof_metrics['n']})", y_ccle.values, ccle_oof, "steelblue"),
        (f"TCGA external (n={tcga_metrics['n']})", y_tcga.values, tcga_proba, "tomato"),
    ]:
        fpr, tpr, _ = roc_curve(y, p)
        auc = roc_auc_score(y, p)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("XGBoost — CCLE within-cohort vs TCGA external")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots / "tcga_xgb_roc.png", dpi=150)
    plt.close()

    # PR
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for label, y, p, color in [
        (f"CCLE OOF", y_ccle.values, ccle_oof, "steelblue"),
        (f"TCGA external", y_tcga.values, tcga_proba, "tomato"),
    ]:
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax.plot(rec, prec, color=color, linewidth=2, label=f"{label}  AP={ap:.3f}")
    ax.axhline(y_tcga.mean(), color="gray", linestyle="--", alpha=0.5,
               label=f"TCGA random ({y_tcga.mean():.3f})")
    ax.axhline(y_ccle.mean(), color="black", linestyle=":", alpha=0.5,
               label=f"CCLE random ({y_ccle.mean():.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("XGBoost — Precision-Recall (CCLE vs TCGA)")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(plots / "tcga_xgb_pr.png", dpi=150)
    plt.close()

    # TCGA confusion matrix
    pred = (tcga_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_tcga.values, pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds", cbar=False,
                xticklabels=["WT", "Mutant"], yticklabels=["WT", "Mutant"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"TCGA — XGBoost (CCLE-trained), n={tcga_metrics['n']}")
    plt.tight_layout()
    plt.savefig(plots / "tcga_xgb_confusion.png", dpi=150)
    plt.close()

    # Per-cancer-type AUC bar
    if not by_type_df.empty:
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(by_type_df))
        ax.bar(x, by_type_df["roc_auc"], color="tomato", edgecolor="white",
               label="ROC-AUC")
        ax.axhline(tcga_metrics["roc_auc"], color="black", linestyle="--",
                   label=f"Pan-TCGA AUC ({tcga_metrics['roc_auc']:.3f})")
        ax.set_xticks(x); ax.set_xticklabels(by_type_df["cancer_type"], rotation=60, ha="right")
        ax.set_ylabel("ROC-AUC"); ax.set_ylim(0, 1.0)
        ax.set_title("XGBoost (CCLE-trained) — TCGA per cancer type (n≥30)")
        for xi, (auc_v, n) in enumerate(zip(by_type_df["roc_auc"], by_type_df["n"])):
            ax.text(xi, auc_v + 0.01, f"n={n}", ha="center", fontsize=7)
        ax.legend()
        plt.tight_layout()
        plt.savefig(plots / "tcga_xgb_per_cancer_type.png", dpi=150)
        plt.close()


if __name__ == "__main__":
    main()
