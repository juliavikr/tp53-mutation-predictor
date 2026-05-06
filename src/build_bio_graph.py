"""Build biological-prior gene-gene graphs from STRING DB physical interactions.

Produces:
  data/processed/gene_graph_bio.npz       — STRING physical PPI restricted to top-2k HVG
  data/processed/gene_graph_hybrid.npz    — union of bio edges + Spearman |rho|>=0.5
  data/processed/graph_overlap.json       — edge counts + bio∩coexp overlap

STRING source: https://string-db.org/cgi/download (v12.0, organism 9606 = human).
We use PHYSICAL interactions only at score >= --score (default 400 = medium confidence).
"""

from __future__ import annotations

import argparse
import gzip
import json
import shutil
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from load_data import PROJECT_ROOT


STRING_BASE = "https://stringdb-static.org/download"
STRING_LINKS = "9606.protein.physical.links.v12.0.txt.gz"
STRING_INFO = "9606.protein.info.v12.0.txt.gz"


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)


def load_string(string_dir: Path, score_threshold: int) -> tuple[pd.DataFrame, dict[str, str]]:
    download(f"{STRING_BASE}/protein.physical.links.v12.0/{STRING_LINKS}",
             string_dir / STRING_LINKS)
    download(f"{STRING_BASE}/protein.info.v12.0/{STRING_INFO}",
             string_dir / STRING_INFO)

    print("Reading STRING info...")
    info = pd.read_csv(string_dir / STRING_INFO, sep="\t", compression="gzip")
    string_to_symbol = dict(zip(info["#string_protein_id"], info["preferred_name"]))
    print(f"  STRING proteins: {len(string_to_symbol):,}")

    print(f"Reading STRING physical links (threshold combined_score >= {score_threshold})...")
    links = pd.read_csv(string_dir / STRING_LINKS, sep=" ", compression="gzip")
    links = links[links["combined_score"] >= score_threshold]
    print(f"  Edges after threshold: {len(links):,}")
    return links, string_to_symbol


def parse_gene_symbol(gene_name: str) -> str:
    """'TSPAN6 (7105)' -> 'TSPAN6'."""
    return gene_name.split(" (")[0]


def filter_to_gene_set(
    links: pd.DataFrame, string_to_symbol: dict[str, str], symbol_to_idx: dict[str, int]
) -> set[tuple[int, int]]:
    """Return set of unordered (i,j) pairs in our gene-set indexing."""
    src_sym = links["protein1"].map(string_to_symbol)
    dst_sym = links["protein2"].map(string_to_symbol)
    src_i = src_sym.map(symbol_to_idx)
    dst_j = dst_sym.map(symbol_to_idx)
    valid = src_i.notna() & dst_j.notna()
    src_i = src_i[valid].astype(int).values
    dst_j = dst_j[valid].astype(int).values
    pairs = set()
    for a, b in zip(src_i, dst_j):
        if a == b:
            continue
        pairs.add((min(a, b), max(a, b)))
    return pairs


def pairs_from_edge_index(edge_index: np.ndarray) -> set[tuple[int, int]]:
    pairs = set()
    for s, d in zip(edge_index[0], edge_index[1]):
        if s == d:
            continue
        pairs.add((min(int(s), int(d)), max(int(s), int(d))))
    return pairs


def pairs_to_edge_index(pairs: set[tuple[int, int]]) -> np.ndarray:
    if not pairs:
        return np.zeros((2, 0), dtype=np.int64)
    iu = np.array([p[0] for p in pairs], dtype=np.int64)
    ju = np.array([p[1] for p in pairs], dtype=np.int64)
    return np.stack([np.concatenate([iu, ju]), np.concatenate([ju, iu])])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed"
    )
    parser.add_argument(
        "--string-dir", type=Path, default=PROJECT_ROOT / "data" / "raw" / "string"
    )
    parser.add_argument("--score", type=int, default=400,
                        help="Minimum STRING combined_score (700=high, 400=medium, 150=low)")
    parser.add_argument(
        "--coexp-graph",
        type=Path,
        default=None,
        help="Path to a Spearman graph .npz to union with for the hybrid graph "
             "(default: data/processed/gene_graph_thr05.npz)",
    )
    args = parser.parse_args()
    proc = args.proc_dir
    coexp_path = args.coexp_graph or (proc / "gene_graph_thr05.npz")

    # Load gene order from the existing co-expression graph (defines our 2k node set)
    coexp_npz = np.load(coexp_path, allow_pickle=False)
    gene_order = coexp_npz["gene_order"].tolist()
    coexp_edge_index = coexp_npz["edge_index"]
    n_genes = len(gene_order)
    print(f"Gene order from {coexp_path.name}: {n_genes} genes")

    symbols = [parse_gene_symbol(g) for g in gene_order]
    symbol_to_idx = {s: i for i, s in enumerate(symbols)}

    links, string_to_symbol = load_string(args.string_dir, args.score)

    bio_pairs = filter_to_gene_set(links, string_to_symbol, symbol_to_idx)
    bio_edge_index = pairs_to_edge_index(bio_pairs)
    print(f"Biological edges in our gene set: {len(bio_pairs):,}")

    coexp_pairs = pairs_from_edge_index(coexp_edge_index)
    print(f"Co-expression edges (from {coexp_path.name}): {len(coexp_pairs):,}")

    intersection = bio_pairs & coexp_pairs
    union = bio_pairs | coexp_pairs
    print(f"Intersection (bio & coexp): {len(intersection):,}")
    print(f"Union      (bio | coexp): {len(union):,}")

    hybrid_edge_index = pairs_to_edge_index(union)

    # Save bio-only graph
    bio_path = proc / "gene_graph_bio.npz"
    np.savez(
        bio_path,
        edge_index=bio_edge_index,
        edge_weight=np.ones(bio_edge_index.shape[1], dtype=np.float32),
        gene_order=np.array(gene_order, dtype=str),
        mode="bio",
        threshold=np.float32(args.score),
        top_k=np.int32(0),
    )
    print(f"Saved → {bio_path}")

    # Save hybrid graph
    hybrid_path = proc / "gene_graph_hybrid.npz"
    np.savez(
        hybrid_path,
        edge_index=hybrid_edge_index,
        edge_weight=np.ones(hybrid_edge_index.shape[1], dtype=np.float32),
        gene_order=np.array(gene_order, dtype=str),
        mode="hybrid",
        threshold=np.float32(args.score),
        top_k=np.int32(0),
    )
    print(f"Saved → {hybrid_path}")

    # Save overlap stats
    overlap = {
        "n_genes": n_genes,
        "string_score_threshold": args.score,
        "coexp_graph": str(coexp_path),
        "bio_edges_undirected": len(bio_pairs),
        "coexp_edges_undirected": len(coexp_pairs),
        "intersection": len(intersection),
        "union": len(union),
        "bio_avg_degree": round(2 * len(bio_pairs) / n_genes, 2),
        "coexp_avg_degree": round(2 * len(coexp_pairs) / n_genes, 2),
        "hybrid_avg_degree": round(2 * len(union) / n_genes, 2),
        "jaccard_similarity": round(len(intersection) / max(len(union), 1), 4),
        "fraction_bio_in_coexp": round(len(intersection) / max(len(bio_pairs), 1), 4),
        "fraction_coexp_in_bio": round(len(intersection) / max(len(coexp_pairs), 1), 4),
    }
    out_json = proc / "graph_overlap.json"
    with open(out_json, "w") as f:
        json.dump(overlap, f, indent=2)
    print(f"\nOverlap stats → {out_json}")
    print(json.dumps(overlap, indent=2))


if __name__ == "__main__":
    main()
