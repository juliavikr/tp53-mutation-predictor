"""XGBoost binary baseline for TP53 mutation status on top-K variable CCLE genes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from load_data import PROJECT_ROOT, load_ccle


def select_top_variable_genes(expr: pd.DataFrame, n_genes: int) -> list[str]:
    return expr.var(axis=0).nlargest(n_genes).index.tolist()


def stratified_cv_splits(y: pd.Series, n_splits: int, seed: int) -> pd.Series:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold = pd.Series(-1, index=y.index, dtype=int)
    for i, (_, test_idx) in enumerate(skf.split(np.zeros(len(y)), y.values)):
        fold.iloc[test_idx] = i
    return fold


def _xgb_clf(seed: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        tree_method="hist",
        random_state=seed,
        n_jobs=-1,
    )


def run_baseline(
    n_genes: int,
    n_splits: int,
    seed: int,
    out_dir: Path,
) -> dict:
    expr, labels = load_ccle()
    y = labels["tp53_binary"]

    top_genes = select_top_variable_genes(expr, n_genes)
    X = expr[top_genes]

    folds = stratified_cv_splits(y, n_splits, seed)
    oof_preds = pd.Series(np.nan, index=y.index, dtype=float)
    fold_metrics: list[dict] = []

    for k in range(n_splits):
        train_mask = folds != k
        test_mask = folds == k

        clf = _xgb_clf(seed)
        clf.fit(X[train_mask].values, y[train_mask].values)
        proba = clf.predict_proba(X[test_mask].values)[:, 1]
        oof_preds[test_mask] = proba

        pred = (proba >= 0.5).astype(int)
        y_test = y[test_mask].values
        fold_metrics.append({
            "fold": k,
            "n_train": int(train_mask.sum()),
            "n_test": int(test_mask.sum()),
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "pr_auc": float(average_precision_score(y_test, proba)),
        })

    metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
    summary = {
        "n_genes": n_genes,
        "n_splits": n_splits,
        "seed": seed,
        "per_fold": fold_metrics,
    }
    for key in metric_keys:
        vals = [m[key] for m in fold_metrics]
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_std"] = float(np.std(vals))

    out_dir.mkdir(parents=True, exist_ok=True)
    folds.to_frame("fold").rename_axis("ModelID").to_csv(out_dir / "cv_splits.csv")
    pd.DataFrame({"tp53_binary": y, "xgb_proba": oof_preds}).to_csv(
        out_dir / "xgb_baseline_oof_preds.csv"
    )
    pd.Series(top_genes, name="gene").to_csv(
        out_dir / "top_genes.csv", index=False
    )
    with open(out_dir / "xgb_baseline_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-genes", type=int, default=2000)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    args = parser.parse_args()

    summary = run_baseline(
        n_genes=args.n_genes,
        n_splits=args.n_splits,
        seed=args.seed,
        out_dir=args.out_dir,
    )
    print(f"Top-{args.n_genes} genes · {args.n_splits}-fold CV")
    for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]:
        m = summary[f"{key}_mean"]
        s = summary[f"{key}_std"]
        print(f"  {key:>10s}: {m:.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
