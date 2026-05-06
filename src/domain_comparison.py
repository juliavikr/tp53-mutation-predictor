"""CCLE vs TCGA domain comparison: PCA, expression distributions, mutation prevalence.

Generates publication-style plots under data/processed/plots/domain/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, load_ccle


def parse_symbol(g: str) -> str:
    return g.split(" (")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--n-pcs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    proc = args.proc_dir
    out = proc / "plots" / "domain"
    out.mkdir(parents=True, exist_ok=True)

    expr_ccle, labels_ccle = load_ccle()
    y_ccle = labels_ccle["tp53_binary"].astype(int)
    expr_tcga = pd.read_csv(proc / "tcga_expression.csv", index_col=0)
    labels_tcga = pd.read_csv(proc / "tcga_labels.csv", index_col=0)
    y_tcga = labels_tcga["tp53_binary"].astype(int)

    top_genes_full = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    symbol_to_full = {parse_symbol(g): g for g in top_genes_full}
    shared = [s for s in symbol_to_full if s in expr_tcga.columns]
    print(f"Shared genes: {len(shared)}")

    X_ccle = expr_ccle[[symbol_to_full[s] for s in shared]].values
    X_tcga_df = expr_tcga[shared].copy()
    n_nan = int(X_tcga_df.isna().sum().sum())
    if n_nan:
        # TCGA Xena matrix has occasional NaNs; impute per-gene median (computed on TCGA).
        X_tcga_df = X_tcga_df.fillna(X_tcga_df.median(numeric_only=True))
        # Any column that was all-NaN -> fill with 0
        X_tcga_df = X_tcga_df.fillna(0.0)
        print(f"  imputed {n_nan} NaNs in TCGA with per-gene medians")
    X_tcga = X_tcga_df.values
    print(f"CCLE: {X_ccle.shape}  TCGA: {X_tcga.shape}")

    # ── 1. Mutation prevalence comparison ──────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["CCLE\n(cell lines)", "TCGA\n(primary tumours)"],
        [y_ccle.mean(), y_tcga.mean()],
        color=["steelblue", "tomato"], edgecolor="white",
    )
    for b, v, n in zip(bars, [y_ccle.mean(), y_tcga.mean()],
                       [len(y_ccle), len(y_tcga)]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{v:.1%}\n(n={n:,})", ha="center", fontsize=11)
    ax.set_ylabel("TP53 mutation prevalence")
    ax.set_ylim(0, 0.75)
    ax.set_title("TP53 mutation prevalence — CCLE cell lines are enriched for TP53 mutations")
    plt.tight_layout()
    plt.savefig(out / "tp53_prevalence.png", dpi=150)
    plt.close()

    # ── 2. Per-cancer-type TP53 prevalence in TCGA ─────────────────────
    by_type = (
        labels_tcga.groupby("cancer_type")["tp53_binary"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "tp53_rate", "count": "n"})
        .query("n >= 30")
        .sort_values("tp53_rate", ascending=False)
    )
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(by_type))
    ax.bar(x, by_type["tp53_rate"], color="tomato", edgecolor="white")
    ax.axhline(y_ccle.mean(), color="steelblue", linestyle="--",
               label=f"CCLE avg ({y_ccle.mean():.2f})")
    ax.axhline(y_tcga.mean(), color="black", linestyle=":",
               label=f"TCGA avg ({y_tcga.mean():.2f})")
    ax.set_xticks(x); ax.set_xticklabels(by_type.index, rotation=60, ha="right")
    ax.set_ylabel("TP53 mutation rate")
    ax.set_ylim(0, 1.0)
    ax.set_title("TP53 mutation prevalence per TCGA cancer type")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out / "tp53_prevalence_per_type.png", dpi=150)
    plt.close()

    # ── 3. Expression distribution of selected TP53 targets ────────────
    target_genes = ["CDKN1A", "MDM2", "BAX", "BBC3", "PHLDA3", "BTG2"]
    available = [g for g in target_genes if g in symbol_to_full and g in expr_tcga.columns]
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, g in zip(axes.flat, available):
        ccle_vals = expr_ccle[symbol_to_full[g]].values
        tcga_vals = expr_tcga[g].values
        ax.hist(ccle_vals, bins=40, color="steelblue", alpha=0.6, label="CCLE", density=True)
        ax.hist(tcga_vals, bins=40, color="tomato", alpha=0.6, label="TCGA", density=True)
        ax.set_title(g)
        ax.set_xlabel("log2(expr+1)")
        ax.legend(fontsize=8)
    plt.suptitle("Expression of selected TP53 pathway genes — CCLE vs TCGA", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "expression_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── 4. PCA on combined CCLE+TCGA (per-cohort z-scored) ────────────
    print("\nRunning PCA on combined cohorts...")
    Xc = StandardScaler().fit_transform(X_ccle)
    Xt = StandardScaler().fit_transform(X_tcga)
    X_all = np.vstack([Xc, Xt])
    cohort = np.array(["CCLE"] * len(Xc) + ["TCGA"] * len(Xt))
    y_all = np.concatenate([y_ccle.values, y_tcga.values])

    pca = PCA(n_components=args.n_pcs, random_state=args.seed)
    Z = pca.fit_transform(X_all)
    print(f"  PC1 var={pca.explained_variance_ratio_[0]*100:.1f}%  "
          f"PC2={pca.explained_variance_ratio_[1]*100:.1f}%  "
          f"top10={pca.explained_variance_ratio_[:10].sum()*100:.1f}%")

    # Combined PCA coloured by cohort
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, c, title in [
        (axes[0], cohort, "by cohort"),
        (axes[1], y_all, "by TP53 status"),
    ]:
        if title == "by cohort":
            for cc, color in zip(["CCLE", "TCGA"], ["steelblue", "tomato"]):
                m = c == cc
                ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, color=color, label=cc, linewidths=0)
        else:
            for v, color, lab in zip([0, 1], ["steelblue", "tomato"], ["WT", "Mutant"]):
                m = c == v
                ax.scatter(Z[m, 0], Z[m, 1], s=4, alpha=0.4, color=color, label=lab, linewidths=0)
        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
        ax.set_title(f"PCA ({title})")
        ax.legend(markerscale=3, fontsize=9)
    plt.suptitle("CCLE + TCGA — PCA on shared top-2k HVG (per-cohort z-score)", y=1.02)
    plt.tight_layout()
    plt.savefig(out / "pca_combined.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ── Summary JSON ───────────────────────────────────────────────────
    summary = {
        "n_shared_genes": len(shared),
        "ccle_n": int(len(y_ccle)),
        "tcga_n": int(len(y_tcga)),
        "ccle_tp53_rate": float(y_ccle.mean()),
        "tcga_tp53_rate": float(y_tcga.mean()),
        "ccle_minus_tcga_rate": float(y_ccle.mean() - y_tcga.mean()),
        "pca_pc1_var": float(pca.explained_variance_ratio_[0]),
        "pca_pc2_var": float(pca.explained_variance_ratio_[1]),
        "pca_top10_var": float(pca.explained_variance_ratio_[:10].sum()),
        "tcga_per_type_top5_pos_rate": by_type.head(5).to_dict(orient="index"),
        "tcga_per_type_bottom5_pos_rate": by_type.tail(5).to_dict(orient="index"),
    }
    with open(proc / "domain_comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary → {proc / 'domain_comparison_summary.json'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
