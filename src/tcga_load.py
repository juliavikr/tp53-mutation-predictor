"""Download + harmonize TCGA pan-cancer expression + TP53 mutation labels.

Two modes:

  DEFAULT — download from UCSC Xena (binary labels only, log2-normalised):
    Source: TCGA PanCancer Atlas hub (https://xenabrowser.net/datapages/)
    Files:
      - EB++AdjustPANCAN_IlluminaHiSeq_RNASeqV2.geneExp.xena.gz
      - mc3.v0.2.8.PUBLIC.nonsilentGene.xena.gz
      - Survival_SupplementalTable_S1_20171025_xena_sp (clinical)

  --from-preprocessed — read from notebooks/04 output (multi-class labels available):
    Source: data/processed/tcga_preprocessed.csv.gz
    Requires notebooks 02–04 to have been executed first.
    Columns expected: tp53_binary (0/1), tp53_class (7-way), Variant_Classification.
    Use this mode for multi-class classification experiments.

Outputs:
  - data/processed/tcga_expression.csv  (samples × top-2k HVG, aligned to CCLE gene order)
  - data/processed/tcga_labels.csv      (sample_id, tp53_binary, cancer_type [, tp53_class])
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


def _load_from_preprocessed(proc: Path, top_genes_path: Path) -> None:
    """Read tcga_preprocessed.csv.gz (notebooks 02-04 output) and write the same
    tcga_expression.csv / tcga_labels.csv / tcga_load_summary.json that the Xena
    path produces, so all downstream scripts work unchanged.

    The preprocessed file is ~18k genes × 9,875 samples; we slice it to the
    top-2k CCLE HVG (same intersection logic as the Xena path).
    """
    preproc = proc / "tcga_preprocessed.csv.gz"
    if not preproc.exists():
        raise FileNotFoundError(
            f"{preproc} not found. Run notebooks 02-04 first, or omit "
            "--from-preprocessed to use the Xena download path."
        )

    print(f"Reading {preproc} ...")
    df = pd.read_csv(preproc, index_col=0, low_memory=False)

    label_cols = ["tp53_binary", "Variant_Classification", "tp53_class"]
    present_labels = [c for c in label_cols if c in df.columns]
    gene_cols = [c for c in df.columns if c not in label_cols]

    top_genes_full = pd.read_csv(top_genes_path)["gene"].tolist()
    top_symbols = [parse_gene_symbol(g) for g in top_genes_full]
    keep = [s for s in top_symbols if s in gene_cols]
    missing = sorted(set(top_symbols) - set(keep))
    print(f"Gene overlap top-2k CCLE ∩ preprocessed TCGA: {len(keep)} / {len(top_symbols)}")
    if missing:
        print(f"  first 10 missing: {missing[:10]}")

    expr_out = df[keep]
    expr_out.index.name = "sample_id"
    expr_out.to_csv(proc / "tcga_expression.csv")

    labels_out = df[present_labels].copy()
    labels_out.index.name = "sample_id"
    # tcga_eval.py expects a 'cancer_type' column; fill with unknown if absent
    if "cancer_type" not in labels_out.columns:
        labels_out["cancer_type"] = "unknown"
    labels_out.to_csv(proc / "tcga_labels.csv")

    summary = {
        "source": "tcga_preprocessed.csv.gz (notebooks 02-04, GDC RSEM)",
        "n_samples": int(len(df)),
        "n_genes_kept": int(len(keep)),
        "n_genes_missing_vs_ccle_top2k": int(len(missing)),
        "tp53_mutant_rate": float(df["tp53_binary"].mean()) if "tp53_binary" in df.columns else None,
        "tp53_class_counts": df["tp53_class"].value_counts().to_dict() if "tp53_class" in df.columns else {},
        "primary_only": True,
    }
    with open(proc / "tcga_load_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("\n=== --from-preprocessed load summary ===")
    print(json.dumps(summary, indent=2))


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
    parser.add_argument("--from-preprocessed", action="store_true", default=False,
                        help="Read from data/processed/tcga_preprocessed.csv.gz (notebooks 02-04 "
                             "output) instead of downloading Xena files. Preserves tp53_class "
                             "multi-class labels. Requires notebooks 02-04 to have been run.")
    args = parser.parse_args()

    proc = args.proc_dir
    tcga = args.tcga_dir
    top_genes_path = args.top_genes or (proc / "top_genes.csv")

    # ── Early-exit path: read from preprocessed notebook output ───────
    if args.from_preprocessed:
        _load_from_preprocessed(proc, top_genes_path)
        return

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
