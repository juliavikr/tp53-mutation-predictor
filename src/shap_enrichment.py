"""Formal pathway enrichment of TP53-pathway genes among top-K SHAP genes.

For each K ∈ {10, 20, 50}, run a one-sided hypergeometric test of the null
"the top-K SHAP genes are a uniform-random sample of the 2,000-gene background".

Reports:
  - K = top-K
  - x = observed pathway hits in top-K
  - K_pop = pathway hits in the 2,000-gene background
  - expected = K * (K_pop / 2000)
  - enrichment_factor = observed / expected
  - hypergeom_p_value (one-sided P[X >= x])
  - fisher_odds_ratio + Fisher's exact p (equivalent formulation)

Outputs:
  - data/processed/shap_enrichment.csv
  - data/processed/plots/shap_enrichment.png  (bar of enrichment factor with p-values)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, hypergeom

from load_data import PROJECT_ROOT
from tp53_pathway import TP53_PATHWAY


def parse_symbol(g: str) -> str:
    return g.split(" (")[0]


def enrichment_test(top_symbols: list[str], background_symbols: list[str],
                    pathway: set[str]) -> dict:
    bg = set(background_symbols)
    pathway_in_bg = pathway & bg
    M = len(bg)               # population size
    K_pop = len(pathway_in_bg)  # successes in population
    n = len(top_symbols)       # sample size
    x = sum(1 for s in top_symbols if s in pathway_in_bg)  # successes in sample

    expected = n * K_pop / M if M else 0.0
    enrichment = (x / expected) if expected else float("nan")

    # One-sided hypergeometric: P(X >= x)
    hg_p = float(hypergeom.sf(x - 1, M, K_pop, n))

    # Fisher exact (one-sided, "greater") — same null, equivalent p-value
    a, b = x, n - x
    c, d = K_pop - x, M - n - K_pop + x
    odds, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")

    return {
        "K_top": n,
        "observed_hits": x,
        "pathway_in_background": K_pop,
        "background_size": M,
        "expected_hits": float(expected),
        "enrichment_factor": float(enrichment),
        "hypergeom_p": hg_p,
        "fisher_odds_ratio": float(odds),
        "fisher_p": float(fisher_p),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--ks", type=int, nargs="+", default=[10, 20, 50])
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    background = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    background_symbols = [parse_symbol(g) for g in background]
    print(f"Background: {len(background_symbols)} top-HVG genes  "
          f"(pathway∩bg = {len(set(background_symbols) & TP53_PATHWAY)} / {len(TP53_PATHWAY)})")

    shap_top = pd.read_csv(proc / "shap_top50.csv")
    shap_symbols_50 = shap_top["gene_symbol"].tolist()
    if len(shap_symbols_50) < max(args.ks):
        raise SystemExit(f"shap_top50.csv has only {len(shap_symbols_50)} rows; "
                         f"need >= {max(args.ks)}")

    rows = []
    for k in args.ks:
        top_k_syms = shap_symbols_50[:k]
        result = enrichment_test(top_k_syms, background_symbols, TP53_PATHWAY)
        result["pathway_hit_symbols"] = ", ".join(
            s for s in top_k_syms if s in TP53_PATHWAY
        )
        rows.append(result)
        print(
            f"\nTop-{k:>2d} SHAP:  {result['observed_hits']}/{k} pathway hits  "
            f"(expected ≈ {result['expected_hits']:.2f})  "
            f"enrichment {result['enrichment_factor']:.2f}×  "
            f"hypergeom p = {result['hypergeom_p']:.2e}  "
            f"Fisher OR = {result['fisher_odds_ratio']:.2f}  p = {result['fisher_p']:.2e}"
        )
        print(f"   hits: {result['pathway_hit_symbols']}")

    df = pd.DataFrame(rows)
    df.to_csv(proc / "shap_enrichment.csv", index=False)
    print(f"\nSaved → {proc / 'shap_enrichment.csv'}")

    # ── Plot ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x_pos = np.arange(len(df))
    bars = ax.bar(
        x_pos, df["enrichment_factor"], color="tomato", edgecolor="white",
    )
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7, label="No enrichment (1×)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Top-{k}\n(n={k})" for k in df["K_top"]], fontsize=10)
    ax.set_ylabel("Enrichment factor (observed / expected)")
    ax.set_title("TP53-pathway enrichment among top-K XGBoost SHAP genes")
    for b, hits, p, ef in zip(bars, df["observed_hits"], df["hypergeom_p"],
                                df["enrichment_factor"]):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2,
                f"{int(hits)} hits\n{ef:.2f}×\np = {p:.1e}",
                ha="center", fontsize=9)
    ax.legend(loc="upper right")
    ax.set_ylim(0, max(df["enrichment_factor"].max() * 1.4, 2.0))
    plt.tight_layout()
    plt.savefig(plots / "shap_enrichment.png", dpi=150)
    plt.close()
    print(f"Saved → {plots / 'shap_enrichment.png'}")


if __name__ == "__main__":
    main()
