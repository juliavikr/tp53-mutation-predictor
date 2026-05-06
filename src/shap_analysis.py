"""SHAP analysis for the XGBoost TP53 binary classifier.

Trains a final XGBoost on all CCLE samples (no CV held-out), computes SHAP values,
and writes:
  - shap_top20.csv     (top-20 genes by mean |SHAP|, annotated with TP53 pathway)
  - shap_top50.csv     (extended for biology look-up)
  - plots/shap_summary.png   (beeswarm)
  - plots/shap_bar.png       (mean |SHAP| ranked)
  - plots/shap_top20_pathway.png  (bar with TP53 pathway membership coloured)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb

from load_data import PROJECT_ROOT, load_ccle
from tp53_pathway import TP53_PATHWAY, annotate
from train_xgb import _xgb_clf


def parse_symbol(gene_name: str) -> str:
    """'TSPAN6 (7105)' -> 'TSPAN6'."""
    return gene_name.split(" (")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    expr, labels = load_ccle()
    y = labels["tp53_binary"]

    top_genes = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    X = expr[top_genes]
    print(f"X: {X.shape}  y_pos={int(y.sum())}/{len(y)}")

    clf = _xgb_clf(args.seed)
    clf.fit(X.values, y.values)
    print("Final XGBoost fitted on full CCLE.")

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X.values)
    print(f"SHAP values: {shap_values.shape}")

    np.save(proc / "shap_values.npy", shap_values.astype(np.float32))

    mean_abs = np.abs(shap_values).mean(axis=0)
    rank = np.argsort(-mean_abs)

    def build_table(n: int) -> pd.DataFrame:
        idx = rank[:n]
        symbols = [parse_symbol(top_genes[i]) for i in idx]
        rows = annotate(symbols)
        out = pd.DataFrame(rows)
        out.insert(0, "rank", np.arange(1, n + 1))
        out.insert(2, "gene_full_name", [top_genes[i] for i in idx])
        out["mean_abs_shap"] = mean_abs[idx]
        # Direction: positive SHAP -> pushes toward "mutant"
        signed_mean = shap_values[:, idx].mean(axis=0)
        out["mean_signed_shap"] = signed_mean
        return out

    top20 = build_table(20)
    top50 = build_table(50)
    top20.to_csv(proc / "shap_top20.csv", index=False)
    top50.to_csv(proc / "shap_top50.csv", index=False)
    pathway_hits = int(top20["in_tp53_pathway"].sum())
    print(f"\nTop-20 SHAP genes: {pathway_hits} are in the curated TP53 pathway set.")
    print(top20[["rank", "gene_symbol", "tp53_category", "mean_abs_shap"]].to_string(index=False))

    # SHAP beeswarm
    plt.figure()
    shap.summary_plot(
        shap_values, X, feature_names=[parse_symbol(g) for g in top_genes],
        max_display=20, show=False,
    )
    plt.tight_layout()
    plt.savefig(plots / "shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()

    # SHAP bar
    plt.figure()
    shap.summary_plot(
        shap_values, X, feature_names=[parse_symbol(g) for g in top_genes],
        max_display=20, plot_type="bar", show=False,
    )
    plt.tight_layout()
    plt.savefig(plots / "shap_bar.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Custom: top-20 with TP53 pathway membership coloured
    fig, ax = plt.subplots(figsize=(8, 7))
    palette = {True: "tomato", False: "steelblue"}
    colors = [palette[m] for m in top20["in_tp53_pathway"]]
    y_pos = np.arange(len(top20))[::-1]
    ax.barh(y_pos, top20["mean_abs_shap"], color=colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([
        f"{r}. {s}  [{c}]" if m else f"{r}. {s}"
        for r, s, c, m in zip(top20["rank"], top20["gene_symbol"],
                              top20["tp53_category"], top20["in_tp53_pathway"])
    ], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Top-20 XGBoost feature importances — TP53 pathway members in red")
    plt.tight_layout()
    plt.savefig(plots / "shap_top20_pathway.png", dpi=150)
    plt.close()

    print(f"\nPathway-set size considered: {len(TP53_PATHWAY)} HGNC symbols")
    print(f"Saved → {proc} (shap_top20.csv, shap_top50.csv, plots/shap_*.png)")


if __name__ == "__main__":
    main()
