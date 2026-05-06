"""Download + harmonize TCGA pan-cancer expression + TP53 mutation labels.

Source: UCSC Xena, TCGA PanCancer Atlas hub (https://xenabrowser.net/datapages/?host=https://pancanatlas.xenahubs.net).

Files:
  - Expression: EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
                (log2(norm_count+1), batch-corrected; ~11k samples × 20k genes; ~3 GB)
  - Mutations : mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz
                (binary nonsilent mutation matrix; ~10k samples × 12k genes)
  - Clinical  : Survival_SupplementalTable_S1_20171025_xena_sp
                (sample type, cancer type — used to filter to primary tumours)

Outputs:
  - data/processed/tcga_expression.csv  (samples × top-2k HVG, aligned to CCLE gene order)
  - data/processed/tcga_labels.csv      (sample_id, tp53_binary, cancer_type)
  - data/processed/tcga_load_summary.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import PROJECT_ROOT


XENA_BASE = "https://pancanatlas.xenahubs.net/download"
EXPR_FILE = "EB%2B%2BAdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz"
MUT_FILE = "mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz"
PHENO_FILE = "Survival_SupplementalTable_S1_20171025_xena_sp"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url}")
    print(f"          -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print(f"          {dest.stat().st_size / 1e6:.1f} MB")


def parse_gene_symbol(gene_name: str) -> str:
    return gene_name.split(" (")[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--tcga-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "tcga"
    )
    parser.add_argument("--top-genes", type=Path, default=None,
                        help="Path to CCLE top_genes.csv (default: <proc>/top_genes.csv)")
    parser.add_argument("--keep-primary-only", action="store_true", default=True,
                        help="Restrict to primary tumour samples (sample type 01).")
    args = parser.parse_args()

    proc = args.proc_dir
    tcga = args.tcga_dir
    top_genes_path = args.top_genes or (proc / "top_genes.csv")

    download(f"{XENA_BASE}/{EXPR_FILE}", tcga / EXPR_FILE)
    download(f"{XENA_BASE}/{MUT_FILE}", tcga / MUT_FILE)
    download(f"{XENA_BASE}/{PHENO_FILE}", tcga / PHENO_FILE)

    # ── Phenotype / sample type ────────────────────────────────────────
    print("\nReading clinical / phenotype file...")
    pheno = pd.read_csv(tcga / PHENO_FILE, sep="\t")
    # Common columns: sample, cancer type abbreviation, sample_type
    sample_col = "sample" if "sample" in pheno.columns else pheno.columns[0]
    pheno = pheno.rename(columns={sample_col: "sample_id"})
    cancer_col = next((c for c in pheno.columns
                       if c.lower() in {"cancer type abbreviation", "type",
                                        "cancer_type"}), None)
    if cancer_col is None:
        cancer_col = pheno.columns[1]
    pheno["cancer_type"] = pheno[cancer_col]
    print(f"  samples in phenotype: {len(pheno):,}")

    # ── Mutation matrix → TP53 status ─────────────────────────────────
    print("\nReading mutation matrix (binary nonsilent)...")
    muts = pd.read_csv(tcga / MUT_FILE, sep="\t", index_col=0, compression="gzip")
    print(f"  shape: {muts.shape}  (genes × samples)")
    if "TP53" not in muts.index:
        raise SystemExit("TP53 not found in mutation matrix index")
    tp53 = muts.loc["TP53"].astype(int)
    tp53.name = "tp53_binary"
    tp53.index.name = "sample_id"
    print(f"  TP53 mutant rate: {tp53.mean():.3f}  (n={len(tp53)})")

    # ── Expression matrix ─────────────────────────────────────────────
    print("\nReading expression matrix (this is ~3 GB, may take a few min)...")
    expr = pd.read_csv(tcga / EXPR_FILE, sep="\t", index_col=0, compression="gzip")
    # Xena expression: rows = genes (HGNC), cols = samples
    print(f"  shape: {expr.shape}  (genes × samples)")

    # ── Restrict to top-2k CCLE HVG ────────────────────────────────────
    top_genes_full = pd.read_csv(top_genes_path)["gene"].tolist()
    top_symbols = [parse_gene_symbol(g) for g in top_genes_full]
    keep_symbols = [s for s in top_symbols if s in expr.index]
    missing = sorted(set(top_symbols) - set(keep_symbols))
    print(f"\nGene overlap CCLE_top2k ∩ TCGA: {len(keep_symbols)} / {len(top_symbols)}")
    if missing:
        print(f"  first 10 missing: {missing[:10]}")

    expr_sub = expr.loc[keep_symbols].T
    expr_sub.index.name = "sample_id"
    print(f"  TCGA expression subset: {expr_sub.shape}  (samples × genes)")

    # ── Common samples (have both expression + mutation) ──────────────
    common = expr_sub.index.intersection(tp53.index)
    print(f"\nSamples with both expression + mutation: {len(common):,}")

    # ── Filter to primary tumours if requested ────────────────────────
    if args.keep_primary_only:
        # TCGA sample IDs end with sample-type code; "01" = primary solid tumour.
        primary_mask = common.str.contains(r"-01[A-Z]?$", regex=True)
        common = common[primary_mask]
        print(f"After restricting to primary tumours: {len(common):,}")

    expr_final = expr_sub.loc[common]
    labels_final = pd.DataFrame({
        "tp53_binary": tp53.loc[common].values,
    }, index=common)

    pheno_idx = pheno.set_index("sample_id")
    labels_final["cancer_type"] = pheno_idx["cancer_type"].reindex(common).values

    # ── Save ──────────────────────────────────────────────────────────
    expr_final.to_csv(proc / "tcga_expression.csv")
    labels_final.to_csv(proc / "tcga_labels.csv")
    summary = {
        "source": "UCSC Xena TCGA PanCancer Atlas hub",
        "expr_file": EXPR_FILE,
        "mut_file": MUT_FILE,
        "n_samples": int(len(common)),
        "n_genes_kept": int(expr_final.shape[1]),
        "n_genes_missing_vs_ccle_top2k": int(len(missing)),
        "tp53_mutant_rate": float(labels_final["tp53_binary"].mean()),
        "primary_only": args.keep_primary_only,
        "cancer_types_top10": labels_final["cancer_type"].value_counts().head(10).to_dict(),
    }
    with open(proc / "tcga_load_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== TCGA load summary ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
