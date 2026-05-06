"""Train a configurable GCN with the XGB-baseline CV splits.

Adds (vs. v1): pos_weight BCE, [expr, zscore] features (per-fold stats),
configurable architecture, train/val split + early stopping on val ROC-AUC,
ReduceLROnPlateau, named runs, training curves saved per fold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from gat import GAT
from gcn import GCN
from load_data import PROJECT_ROOT, load_ccle


def load_graph(path: Path) -> tuple[torch.Tensor, list[str]]:
    npz = np.load(path, allow_pickle=False)
    edge_index = torch.tensor(npz["edge_index"], dtype=torch.long)
    gene_order = npz["gene_order"].tolist()
    return edge_index, gene_order


def fit_features(
    X_train: np.ndarray, X_full: np.ndarray, feature_set: str
) -> np.ndarray:
    """Compute features per-fold using only training-fold statistics. Returns (n, g, n_feat)."""
    if feature_set == "expr":
        return X_full[:, :, None]
    if feature_set == "expr_zscore":
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-6
        z = (X_full - mean) / std
        return np.stack([X_full, z], axis=-1).astype(np.float32)
    raise ValueError(f"unknown feature set: {feature_set}")


def make_data(
    x_row: np.ndarray, y_val: float, edge_index: torch.Tensor
) -> Data:
    x = torch.from_numpy(x_row.astype(np.float32))
    if x.dim() == 1:
        x = x.unsqueeze(-1)
    return Data(x=x, edge_index=edge_index, y=torch.tensor([y_val], dtype=torch.float32))


def train_one_epoch(model, loader, optimizer, loss_fn, device) -> float:
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
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    probs, ys, losses, total = [], [], 0.0, 0
    bce = torch.nn.BCEWithLogitsLoss(reduction="sum")
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        losses += bce(out, batch.y).item()
        total += batch.num_graphs
        probs.append(torch.sigmoid(out).cpu().numpy())
        ys.append(batch.y.cpu().numpy())
    return np.concatenate(probs), np.concatenate(ys), losses / total


def run(
    graph_file: Path,
    run_name: str,
    feature_set: str,
    n_layers: int,
    hidden_dim: int,
    dropout: float,
    use_batch_norm: bool,
    use_residual: bool,
    pos_weight_mode: str,
    max_epochs: int,
    patience: int,
    val_frac: float,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    proc_dir: Path,
    model_kind: str = "gcn",
    gat_heads: int = 4,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    expr, labels = load_ccle()
    y = labels["tp53_binary"]

    edge_index, gene_order = load_graph(graph_file)
    folds = pd.read_csv(proc_dir / "cv_splits.csv", index_col="ModelID")["fold"]

    common = expr.index.intersection(folds.index)
    expr = expr.loc[common, gene_order]
    y = y.loc[common]
    folds = folds.loc[common]
    print(f"Aligned: {len(common)} cell lines × {len(gene_order)} genes  (graph={graph_file.name})")

    X = expr.values.astype(np.float32)
    y_vals = y.values.astype(np.float32)
    in_dim = 1 if feature_set == "expr" else 2

    n_splits = int(folds.max()) + 1
    oof_preds = pd.Series(np.nan, index=common, dtype=float)
    fold_metrics: list[dict] = []
    all_curves: list[dict] = []

    for k in range(n_splits):
        train_ids = folds[folds != k].index.tolist()
        test_ids = folds[folds == k].index.tolist()

        train_y = y.loc[train_ids].values
        sub_train_ids, sub_val_ids = train_test_split(
            train_ids,
            test_size=val_frac,
            stratify=train_y,
            random_state=seed + k,
        )
        sub_train_pos = pd.Index(common).get_indexer(sub_train_ids)
        sub_val_pos = pd.Index(common).get_indexer(sub_val_ids)
        test_pos = pd.Index(common).get_indexer(test_ids)

        X_train_fold = X[sub_train_pos]
        X_feat = fit_features(X_train_fold, X, feature_set)

        data_by_id = {
            mid: make_data(X_feat[i], y_vals[i], edge_index)
            for i, mid in enumerate(common)
        }
        train_set = [data_by_id[m] for m in sub_train_ids]
        val_set = [data_by_id[m] for m in sub_val_ids]
        test_set = [data_by_id[m] for m in test_ids]

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        if model_kind == "gat":
            model = GAT(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                heads=gat_heads,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual,
            ).to(device)
        else:
            model = GCN(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                n_layers=n_layers,
                dropout=dropout,
                use_batch_norm=use_batch_norm,
                use_residual=use_residual,
            ).to(device)

        if pos_weight_mode == "balanced":
            n_pos = int((y_vals[sub_train_pos] == 1).sum())
            n_neg = int((y_vals[sub_train_pos] == 0).sum())
            pw = float(n_neg) / max(n_pos, 1)
            pos_weight_tensor = torch.tensor([pw], dtype=torch.float32, device=device)
        elif pos_weight_mode == "none":
            pos_weight_tensor = None
        else:
            pos_weight_tensor = torch.tensor(
                [float(pos_weight_mode)], dtype=torch.float32, device=device
            )

        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-6
        )

        best_auc = -1.0
        best_state = None
        bad = 0
        stop_epoch = max_epochs

        for epoch in range(1, max_epochs + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
            val_proba, val_y, val_loss = evaluate(model, val_loader, device)
            val_auc = float(roc_auc_score(val_y, val_proba))
            val_pred = (val_proba >= 0.5).astype(int)
            val_f1 = float(f1_score(val_y, val_pred, zero_division=0))
            current_lr = optimizer.param_groups[0]["lr"]

            scheduler.step(val_auc)

            all_curves.append({
                "run": run_name,
                "fold": k,
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "val_f1": val_f1,
                "lr": current_lr,
            })

            improved = val_auc > best_auc + 1e-4
            if improved:
                best_auc = val_auc
                best_state = {kk: vv.detach().cpu().clone() for kk, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1

            if epoch == 1 or epoch % 20 == 0 or improved:
                print(
                    f"  fold {k}  ep {epoch:>3d}  loss {train_loss:.4f}  "
                    f"val_loss {val_loss:.4f}  val_auc {val_auc:.4f}  "
                    f"val_f1 {val_f1:.4f}  lr {current_lr:.2e}"
                    + (" *" if improved else "")
                )

            if bad >= patience:
                stop_epoch = epoch
                print(f"  fold {k}  early stop at epoch {epoch}  (best val_auc={best_auc:.4f})")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        proba, y_test, _ = evaluate(model, test_loader, device)
        oof_preds.loc[test_ids] = proba

        pred = (proba >= 0.5).astype(int)
        fold_metrics.append({
            "fold": k,
            "n_train": len(sub_train_ids),
            "n_val": len(sub_val_ids),
            "n_test": len(test_ids),
            "stop_epoch": stop_epoch,
            "best_val_auc": best_auc,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })
        m = fold_metrics[-1]
        print(
            f"fold {k}  test  AUC={m['roc_auc']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
            f"F1={m['f1']:.4f}  Acc={m['accuracy']:.4f}"
        )

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = {
        "run_name": run_name,
        "model_kind": model_kind,
        "graph_file": str(graph_file),
        "feature_set": feature_set,
        "n_layers": n_layers,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "gat_heads": gat_heads if model_kind == "gat" else None,
        "use_batch_norm": use_batch_norm,
        "use_residual": use_residual,
        "pos_weight_mode": pos_weight_mode,
        "max_epochs": max_epochs,
        "patience": patience,
        "val_frac": val_frac,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "n_splits": n_splits,
        "per_fold": fold_metrics,
    }
    for key in metric_keys:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    prefix = f"gcn_{run_name}_" if run_name else "gcn_"
    pd.DataFrame({"tp53_binary": y, f"{prefix}proba".rstrip("_") + "_proba": oof_preds}).to_csv(
        proc_dir / f"{prefix}oof_preds.csv"
    )
    pd.DataFrame(all_curves).to_csv(proc_dir / f"{prefix}curves.csv", index=False)
    with open(proc_dir / f"{prefix}metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-file", type=Path, required=True)
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--feature-set", choices=["expr", "expr_zscore"], default="expr_zscore")
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--batch-norm", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--pos-weight", default="balanced",
                        help="'none', 'balanced', or a float")
    parser.add_argument("--max-epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--model-kind", choices=["gcn", "gat"], default="gcn")
    parser.add_argument("--gat-heads", type=int, default=4)
    args = parser.parse_args()

    summary = run(
        graph_file=args.graph_file,
        run_name=args.run_name,
        feature_set=args.feature_set,
        n_layers=args.n_layers,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_batch_norm=args.batch_norm,
        use_residual=args.residual,
        pos_weight_mode=args.pos_weight,
        max_epochs=args.max_epochs,
        patience=args.patience,
        val_frac=args.val_frac,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        proc_dir=args.proc_dir,
        model_kind=args.model_kind,
        gat_heads=args.gat_heads,
    )
    print(f"\n{args.model_kind.upper()} run='{args.run_name}'  graph={args.graph_file.name}  feat={args.feature_set}")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        m = summary[f"{key}_mean"]
        s = summary[f"{key}_std"]
        print(f"  {key:>10s}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
