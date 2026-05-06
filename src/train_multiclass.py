"""Task 2 — Multiclass TP53 mutation-type classification on CCLE.

Subset: TP53-mutant cell lines only (n ≈ 986).
Classes (after merging rare types — see note below):
  - Missense   (Missense_Mutation)
  - Truncating (Frame_Shift_Del ∪ Splice_Site)
  - Other      (everything else still flagged as TP53-mutant)

Class-merge rule: Frame_Shift_Del (n≈48) and Splice_Site (n≈65) are individually too small
for stable 5-fold CV; both are functionally truncating events so we merge them. The "Other"
bucket is kept to absorb e.g. nonsense / in-frame / intronic edge cases.

Models:
  - XGBoost  (multi:softprob)
  - Logistic Regression (multinomial, L2)

Outputs (data/processed/):
  - multiclass_class_distribution.csv
  - multiclass_metrics.json
  - multiclass_per_class_metrics.csv
  - multiclass_oof_preds.csv
  - top_genes_multiclass.csv             (top-K genes per class by XGB feature importance)
  - plots/multiclass_confusion_*.png
  - plots/multiclass_per_class_f1.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from load_data import PROJECT_ROOT, load_ccle


CLASSES = ["Missense", "Truncating", "Other"]


def merge_classes(s: pd.Series) -> pd.Series:
    mapping = {
        "Missense_Mutation": "Missense",
        "Frame_Shift_Del": "Truncating",
        "Splice_Site": "Truncating",
        "Other": "Other",
    }
    return s.map(mapping).fillna("Other")


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str]) -> pd.DataFrame:
    p = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    f = f1_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    support = np.array([(y_true == c).sum() for c in classes])
    return pd.DataFrame({
        "class": classes, "precision": p, "recall": r, "f1": f, "support": support,
    })


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray,
             classes: list[str]) -> dict:
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }
    # OvR macro AUC
    try:
        out["roc_auc_ovr_macro"] = float(roc_auc_score(
            pd.get_dummies(pd.Series(y_true), columns=classes).reindex(
                columns=classes, fill_value=0).values,
            y_proba, multi_class="ovr", average="macro",
        ))
    except Exception:
        out["roc_auc_ovr_macro"] = float("nan")
    return out


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, classes: list[str],
                    name: str, out_path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=axes[0])
    axes[0].set_title(f"{name} — counts")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", cbar=False,
                vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, ax=axes[1])
    axes[1].set_title(f"{name} — row-normalised")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROJECT_ROOT / "data" / "processed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--top-k-genes", type=int, default=15,
                        help="Top-K genes per class to save (XGB feature importance)")
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    expr, labels = load_ccle()
    is_mut = labels["tp53_binary"] == 1
    expr = expr[is_mut]
    y_raw = labels.loc[is_mut, "tp53_class"]
    y = merge_classes(y_raw)
    print(f"TP53-mutant cell lines: {len(y)}")
    dist = y.value_counts().reindex(CLASSES, fill_value=0)
    print("\nClass distribution (after merge):")
    print(dist.to_string())
    dist_df = dist.rename_axis("class").reset_index(name="count")
    dist_df["fraction"] = dist_df["count"] / dist_df["count"].sum()
    dist_df.to_csv(proc / "multiclass_class_distribution.csv", index=False)

    top_genes_full = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    X = expr[top_genes_full].values.astype(np.float32)
    print(f"\nFeatures: {X.shape}  (top-2k HVG from binary task)")

    le = {c: i for i, c in enumerate(CLASSES)}
    y_int = y.map(le).values

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_proba = {m: np.zeros((len(y), len(CLASSES))) for m in ["xgb", "logreg"]}
    fold_metrics: dict[str, list[dict]] = {m: [] for m in oof_proba}
    importance_per_class = np.zeros((len(CLASSES), X.shape[1]))

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_int)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # XGBoost (multi:softprob)
        xgb_clf = xgb.XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob", num_class=len(CLASSES),
            tree_method="hist", random_state=args.seed, n_jobs=-1,
        )
        xgb_clf.fit(X_train, y_train)
        oof_proba["xgb"][test_idx] = xgb_clf.predict_proba(X_test)

        # Per-class importance via one-vs-rest XGB on the train fold
        # (gain-based importance from a binary model per class).
        for c, name in enumerate(CLASSES):
            y_bin = (y_train == c).astype(int)
            if y_bin.sum() < 5:
                continue
            ovr = xgb.XGBClassifier(
                n_estimators=200, max_depth=5, learning_rate=0.05,
                tree_method="hist", random_state=args.seed, n_jobs=-1,
                objective="binary:logistic",
            )
            ovr.fit(X_train, y_bin)
            importance_per_class[c] += ovr.feature_importances_

        # Logistic Regression (multinomial, L2, scaled)
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)
        lr_clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=1000,
            n_jobs=-1, random_state=args.seed,
        )
        lr_clf.fit(Xs_train, y_train)
        oof_proba["logreg"][test_idx] = lr_clf.predict_proba(Xs_test)

        # Fold metrics for reporting
        for m, P in oof_proba.items():
            y_pred = P[test_idx].argmax(axis=1)
            fm = {
                "fold": fold,
                "accuracy": accuracy_score(y_test, y_pred),
                "macro_f1": f1_score(y_test, y_pred, average="macro", zero_division=0),
                "weighted_f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
            }
            fold_metrics[m].append(fm)
        print(f"  fold {fold}: "
              f"xgb_acc={fold_metrics['xgb'][-1]['accuracy']:.3f}  "
              f"xgb_macroF1={fold_metrics['xgb'][-1]['macro_f1']:.3f}  "
              f"lr_acc={fold_metrics['logreg'][-1]['accuracy']:.3f}  "
              f"lr_macroF1={fold_metrics['logreg'][-1]['macro_f1']:.3f}")

    # ── Aggregate (OOF) ────────────────────────────────────────────────
    summary: dict = {
        "n_samples": int(len(y)),
        "classes": CLASSES,
        "class_distribution": dist.to_dict(),
        "n_features": int(X.shape[1]),
        "n_splits": args.n_splits,
        "merge_rule": "Frame_Shift_Del + Splice_Site -> Truncating",
        "models": {},
    }
    per_class_rows = []
    for m in oof_proba:
        proba = oof_proba[m]
        y_pred = proba.argmax(axis=1)
        y_true_lbl = pd.Series(y_int).map({i: c for c, i in le.items()}).values
        y_pred_lbl = pd.Series(y_pred).map({i: c for c, i in le.items()}).values

        agg = evaluate(y_true_lbl, y_pred_lbl, proba, CLASSES)
        agg["per_fold"] = fold_metrics[m]
        summary["models"][m] = agg

        pcm = per_class_metrics(y_true_lbl, y_pred_lbl, CLASSES)
        pcm["model"] = m
        per_class_rows.append(pcm)

        plot_confusion(y_true_lbl, y_pred_lbl, CLASSES,
                        m.upper(), plots / f"multiclass_confusion_{m}.png")

    pd.concat(per_class_rows, ignore_index=True).to_csv(
        proc / "multiclass_per_class_metrics.csv", index=False
    )

    # OOF predictions
    oof_df_parts = []
    for m, proba in oof_proba.items():
        df = pd.DataFrame(proba, columns=[f"{m}_{c}" for c in CLASSES],
                          index=expr.index)
        oof_df_parts.append(df)
    oof_df = pd.concat([
        pd.DataFrame({"true_class": y.values}, index=expr.index),
        *oof_df_parts,
    ], axis=1)
    oof_df.index.name = "ModelID"
    oof_df.to_csv(proc / "multiclass_oof_preds.csv")

    # ── Per-class top-K genes (XGBoost OvR aggregated importance) ──────
    importance_per_class /= args.n_splits  # mean across folds
    top_rows = []
    for c, name in enumerate(CLASSES):
        order = np.argsort(-importance_per_class[c])[:args.top_k_genes]
        for rank, idx in enumerate(order, start=1):
            gene_full = top_genes_full[idx]
            top_rows.append({
                "class": name, "rank": rank,
                "gene_full": gene_full,
                "gene_symbol": gene_full.split(" (")[0],
                "mean_xgb_importance": float(importance_per_class[c, idx]),
            })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(proc / "top_genes_multiclass.csv", index=False)

    with open(proc / "multiclass_metrics.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Plot — per-class F1 bar ─────────────────────────────────────────
    pcm_xgb = per_class_rows[0]; pcm_lr = per_class_rows[1]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    width = 0.4
    x_pos = np.arange(len(CLASSES))
    ax.bar(x_pos - width / 2, pcm_xgb["f1"], width, color="steelblue",
           edgecolor="white", label="XGBoost")
    ax.bar(x_pos + width / 2, pcm_lr["f1"], width, color="tomato",
           edgecolor="white", label="LogReg")
    for i, (a, b, support) in enumerate(zip(pcm_xgb["f1"], pcm_lr["f1"],
                                             pcm_xgb["support"])):
        ax.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i, -0.05, f"n={support}", ha="center", fontsize=8, color="gray")
    ax.set_xticks(x_pos); ax.set_xticklabels(CLASSES)
    ax.set_ylim(-0.08, 1.0); ax.set_ylabel("Per-class F1 (OOF)")
    ax.set_title("Task 2 — Per-class F1, TP53 mutation-type classification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots / "multiclass_per_class_f1.png", dpi=150)
    plt.close()

    # ── Print final summary ────────────────────────────────────────────
    print("\n=== Multiclass OOF results (n=%d, %d classes) ===" % (len(y), len(CLASSES)))
    for m in oof_proba:
        a = summary["models"][m]
        print(f"  {m:>6s}:  acc={a['accuracy']:.4f}  macroF1={a['macro_f1']:.4f}  "
              f"weightedF1={a['weighted_f1']:.4f}  AUC_OvR={a['roc_auc_ovr_macro']:.4f}")
    print("\nPer-class metrics:")
    print(pd.concat(per_class_rows, ignore_index=True).round(4).to_string(index=False))

    print(f"\nTop-{args.top_k_genes} genes per class saved → {proc / 'top_genes_multiclass.csv'}")
    print(f"All artefacts → {proc}")


if __name__ == "__main__":
    main()
