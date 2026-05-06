"""Build the shared gene-gene co-expression graph from CCLE top-K HVG.

Two construction modes:
  threshold : keep undirected edges where |Spearman rho| >= --threshold
  topk      : per gene, keep the top --top-k strongest correlations (then symmetrise)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import PROJECT_ROOT, load_expression


def top_variable_genes(expr: pd.DataFrame, n_genes: int) -> list[str]:
    return expr.var(axis=0).nlargest(n_genes).index.tolist()


def spearman_corr(expr_subset: pd.DataFrame) -> np.ndarray:
    ranked = expr_subset.rank(axis=0).values
    return np.corrcoef(ranked, rowvar=False)


def build_threshold_edges(
    corr: np.ndarray, threshold: float
) -> tuple[np.ndarray, np.ndarray]:
    n = corr.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    mask = np.abs(corr[iu, ju]) >= threshold
    src = iu[mask]
    dst = ju[mask]
    w = corr[src, dst].astype(np.float32)
    edge_index = np.stack(
        [np.concatenate([src, dst]), np.concatenate([dst, src])]
    ).astype(np.int64)
    edge_weight = np.concatenate([w, w])
    return edge_index, edge_weight


def build_topk_edges(corr: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    n = corr.shape[0]
    abs_corr = np.abs(corr).copy()
    np.fill_diagonal(abs_corr, -1.0)
    # argpartition gives unsorted top-k indices per row in O(n)
    top_idx = np.argpartition(-abs_corr, k, axis=1)[:, :k]
    src = np.repeat(np.arange(n), k)
    dst = top_idx.flatten()
    # Symmetrise: keep unordered pairs (min, max), then deduplicate.
    pair = np.sort(np.stack([src, dst]), axis=0).T  # (n*k, 2)
    pair = np.unique(pair, axis=0)
    iu, ju = pair[:, 0], pair[:, 1]
    edge_index = np.stack(
        [np.concatenate([iu, ju]), np.concatenate([ju, iu])]
    ).astype(np.int64)
    w = corr[iu, ju].astype(np.float32)
    edge_weight = np.concatenate([w, w])
    return edge_index, edge_weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--mode", choices=["threshold", "topk"], default="threshold")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--name",
        type=str,
        default="",
        help="Suffix for output files (e.g. 'thr0.7' -> gene_graph_thr0.7.npz). Empty = gene_graph.npz.",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--save-corr",
        action="store_true",
        help="Save the full Spearman correlation matrix (~16 MB for 2k genes).",
    )
    args = parser.parse_args()

    expr = load_expression()
    print(f"Expression matrix : {expr.shape}")

    genes = top_variable_genes(expr, args.n_genes)
    print(f"Selected top {len(genes)} variable genes")

    corr = spearman_corr(expr[genes])
    print(f"Spearman correlation matrix: {corr.shape}")

    if args.mode == "threshold":
        edge_index, edge_weight = build_threshold_edges(corr, args.threshold)
        spec = f"|rho| >= {args.threshold}"
    else:
        edge_index, edge_weight = build_topk_edges(corr, args.top_k)
        spec = f"top-{args.top_k} per gene"

    n_undirected = edge_index.shape[1] // 2
    avg_degree = edge_index.shape[1] / args.n_genes
    print(f"Mode     : {args.mode}  ({spec})")
    print(f"Edges    : {n_undirected:,} undirected ({edge_index.shape[1]:,} directed)")
    print(f"Avg deg  : {avg_degree:.1f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.name}" if args.name else ""
    out_path = args.out_dir / f"gene_graph{suffix}.npz"
    np.savez(
        out_path,
        edge_index=edge_index,
        edge_weight=edge_weight,
        gene_order=np.array(genes, dtype=str),
        mode=args.mode,
        threshold=np.float32(args.threshold),
        top_k=np.int32(args.top_k),
    )
    print(f"Saved graph → {out_path}")

    if args.save_corr:
        corr_path = args.out_dir / f"spearman_corr{suffix}.npz"
        np.savez(
            corr_path,
            corr=corr.astype(np.float32),
            gene_order=np.array(genes, dtype=str),
        )
        print(f"Saved Spearman matrix → {corr_path}  ({corr.nbytes / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
