"""External validation of best GCN/GAT on TCGA primary tumours.

Trains a model on the FULL CCLE cohort (with a small internal val split for early
stopping), then applies it to TCGA samples. Per-cohort z-score normalisation is
used (CCLE for training, TCGA for inference) — same protocol as the XGBoost
external-validation script.

Genes missing in TCGA (149 of the CCLE top-2k) are padded with zeros in TCGA
node features; the graph topology is unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
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
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gat import GAT
from gcn import GCN
from load_data import PROJECT_ROOT, load_ccle


def parse_symbol(g: str) -> str:
    return g.split(" (")[0]


def load_graph(path: Path) -> tuple[torch.Tensor, list[str]]:
    npz = np.load(path, allow_pickle=False)
    edge_index = torch.tensor(npz["edge_index"], dtype=torch.long)
    return edge_index, npz["gene_order"].tolist()


def make_data(x_row: np.ndarray, y_val: float, edge_index: torch.Tensor) -> Data:
    x = torch.from_numpy(x_row.astype(np.float32))
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return Data(x=x, edge_index=edge_index,
                y=torch.tensor([y_val], dtype=torch.float32))


def fit_features(X_train: np.ndarray, X_full: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == "expr":
        return X_full[:, :, None].astype(np.float32)
    if feature_set == "expr_zscore":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-6
        z = (X_full - mean) / std
        return np.stack([X_full, z], axis=-1).astype(np.float32)
    raise ValueError(feature_set)


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


def train_one(model, loader, optimizer, loss_fn, device) -> float:
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total += loss.item() * batch.num_graphs
        n += batch.num_graphs
    return total / n


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, ys = [], []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        probs.append(torch.sigmoid(out).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-file", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="tcga_eval")
    parser.add_argument("--model-kind", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--feature-set", choices=["expr", "expr_zscore"],
                        default="expr_zscore")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--gat-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proc-dir", type=Path,
                        default=PROJECT_ROOT / "data" / "processed")
    args = parser.parse_args()

    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    expr_ccle, labels_ccle = load_ccle()
    y_ccle_full = labels_ccle["tp53_binary"].astype(int)
    edge_index, gene_order = load_graph(args.graph_file)
    print(f"Graph: {args.graph_file.name}  {len(gene_order)} genes  "
          f"{edge_index.shape[1] // 2} undirected edges")

    # Align CCLE expression to graph gene order
    X_ccle = expr_ccle[gene_order].values.astype(np.float32)
    print(f"CCLE: {X_ccle.shape}")

    # Internal CCLE train/val split for early stopping
    train_ids, val_ids = train_test_split(
        np.arange(len(y_ccle_full)),
        test_size=args.val_frac,
        stratify=y_ccle_full.values,
        random_state=args.seed,
    )

    # Z-score using CCLE training fold only
    X_ccle_feat = fit_features(X_ccle[train_ids], X_ccle, args.feature_set)
    print(f"CCLE features: {X_ccle_feat.shape}  (in_dim={X_ccle_feat.shape[-1]})")

    train_set = [make_data(X_ccle_feat[i], float(y_ccle_full.values[i]), edge_index)
                 for i in train_ids]
    val_set = [make_data(X_ccle_feat[i], float(y_ccle_full.values[i]), edge_index)
               for i in val_ids]
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    in_dim = X_ccle_feat.shape[-1]
    if args.model_kind == "gat":
        model = GAT(in_dim=in_dim, hidden_dim=args.hidden_dim,
                    n_layers=args.n_layers, heads=args.gat_heads,
                    dropout=args.dropout, use_batch_norm=True,
                    use_residual=True).to(device)
    else:
        model = GCN(in_dim=in_dim, hidden_dim=args.hidden_dim,
                    n_layers=args.n_layers, dropout=args.dropout,
                    use_batch_norm=True, use_residual=True).to(device)

    n_pos = int((y_ccle_full.values[train_ids] == 1).sum())
    n_neg = int((y_ccle_full.values[train_ids] == 0).sum())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
    )

    best_auc, best_state, bad = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        train_loss = train_one(model, train_loader, optimizer, loss_fn, device)
        val_proba, val_y = predict(model, val_loader, device)
        val_auc = float(roc_auc_score(val_y, val_proba))
        scheduler.step(val_auc)
        if val_auc > best_auc + 1e-4:
            best_auc, best_state, bad = val_auc, {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }, 0
            improved = True
        else:
            bad += 1
            improved = False
        if epoch == 1 or epoch % 10 == 0 or improved:
            print(f"  ep {epoch:>3d}  loss {train_loss:.4f}  val_auc {val_auc:.4f}"
                  + (" *" if improved else ""))
        if bad >= args.patience:
            print(f"  early stop at ep {epoch}  (best val_auc={best_auc:.4f})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    # ── TCGA inference ──────────────────────────────────────────────────
    print("\nTCGA inference...")
    expr_tcga = pd.read_csv(proc / "tcga_expression.csv", index_col=0)
    labels_tcga = pd.read_csv(proc / "tcga_labels.csv", index_col=0)
    y_tcga = labels_tcga["tp53_binary"].astype(int).values

    # Align TCGA to graph gene order; fill missing with 0
    tcga_aligned = pd.DataFrame(0.0, index=expr_tcga.index, columns=gene_order)
    symbol_to_full = {parse_symbol(g): g for g in gene_order}
    for sym in expr_tcga.columns:
        if sym in symbol_to_full:
            tcga_aligned[symbol_to_full[sym]] = expr_tcga[sym].fillna(0).values
    n_present = sum(1 for s in expr_tcga.columns if s in symbol_to_full)
    print(f"  TCGA aligned: {tcga_aligned.shape}  ({n_present}/{len(gene_order)} graph nodes have TCGA data)")

    X_tcga = tcga_aligned.values.astype(np.float32)
    # TCGA features: per-cohort z-score (own stats; matches what we did in XGB)
    X_tcga_feat = fit_features(X_tcga, X_tcga, args.feature_set)

    tcga_set = [
        make_data(X_tcga_feat[i], float(y_tcga[i]), edge_index)
        for i in range(len(y_tcga))
    ]
    tcga_loader = DataLoader(tcga_set, batch_size=args.batch_size, shuffle=False)
    tcga_proba, _ = predict(model, tcga_loader, device)

    tcga_metrics = metrics_from(y_tcga, tcga_proba)
    print("\nTCGA external metrics:")
    for k, v in tcga_metrics.items():
        print(f"  {k:>14s}: {v:.4f}" if isinstance(v, float) else f"  {k:>14s}: {v}")

    # Per cancer type
    by_type = []
    for ct, sub in labels_tcga.groupby("cancer_type"):
        if len(sub) < 30:
            continue
        idx = labels_tcga.index.get_indexer(sub.index)
        m = metrics_from(y_tcga[idx], tcga_proba[idx])
        m["cancer_type"] = ct
        by_type.append(m)
    by_type_df = pd.DataFrame(by_type).sort_values("roc_auc", ascending=False)

    # Save predictions + summary + plots
    out_prefix = f"tcga_{args.model_kind}_{args.run_name}"
    pd.DataFrame({
        "sample_id": labels_tcga.index,
        "tp53_binary": y_tcga,
        "cancer_type": labels_tcga["cancer_type"].values,
        f"{args.model_kind}_proba": tcga_proba,
    }).to_csv(proc / f"{out_prefix}_oof_preds.csv", index=False)
    by_type_df.to_csv(proc / f"{out_prefix}_per_cancer_type.csv", index=False)

    summary = {
        "model_kind": args.model_kind,
        "run_name": args.run_name,
        "graph_file": str(args.graph_file),
        "feature_set": args.feature_set,
        "n_layers": args.n_layers,
        "hidden_dim": args.hidden_dim,
        "best_val_auc_ccle": best_auc,
        "tcga_metrics": tcga_metrics,
    }
    with open(proc / f"{out_prefix}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary → {proc / f'{out_prefix}_summary.json'}")

    # Plot: ROC + per-cancer
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    fpr, tpr, _ = roc_curve(y_tcga, tcga_proba)
    auc = roc_auc_score(y_tcga, tcga_proba)
    ax.plot(fpr, tpr, color="darkorange", linewidth=2,
            label=f"{args.model_kind.upper()} on TCGA (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"TCGA external validation — {args.model_kind.upper()} ({args.run_name})")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(plots / f"{out_prefix}_roc.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
