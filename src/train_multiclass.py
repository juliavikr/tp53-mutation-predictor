"""Task 2 — Multiclass TP53 mutation-type classification on CCLE.

Subset: TP53-mutant cell lines only (WT excluded).
Silent/synonymous mutations excluded from positive class (treated as WT-like).

Classes (fine-grained, biologically motivated):
  - Missense   : missense_variant — dominant-negative or gain-of-function p53
  - Nonsense   : stop_gained / start_lost — premature termination, LOF
  - Frameshift : frameshift_variant (del + ins merged) — LOF via reading-frame disruption
  - Splice     : splice_donor/acceptor — aberrant splicing, LOF
  - InFrame    : inframe_insertion/deletion — altered protein, partial function

Additional modes (--mode flag):
  - hotspot   : Hotspot Missense vs Non-hotspot Missense (missense only)
  - binary_ms : Missense vs WT binary (exclude all other mutant subtypes)

Models:
  - XGBoost  (multi:softprob)
  - Logistic Regression (multinomial, L2)

Outputs (data/processed/):
  - multiclass_class_distribution.csv
  - multiclass_metrics.json
  - multiclass_per_class_metrics.csv
  - multiclass_oof_preds.csv
  - top_genes_multiclass.csv
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


# Fine-grained classes (biologically distinct mechanisms)
CLASSES = ["Missense", "Nonsense", "Frameshift", "Splice", "InFrame"]

# Minimum per-class samples required for stable 5-fold CV
MIN_CLASS_SIZE = 10


def merge_classes(s: pd.Series) -> pd.Series:
    """Map fine-grained tp53_class to training classes.

    Drops rows where class is WT or silent-like (returns NaN → caller filters).
    InFrame is kept separate if large enough; otherwise callers may merge into Other.
    """
    mapping = {
        "Missense":   "Missense",
        "Nonsense":   "Nonsense",
        "Frameshift": "Frameshift",
        "Splice":     "Splice",
        "InFrame":    "InFrame",
        # Old-style names from previous load_data version — keep backward compat
        "Missense_Mutation": "Missense",
        "Frame_Shift_Del":   "Frameshift",
        "Splice_Site":       "Splice",
        "Other":             None,   # drop; too heterogeneous
        "WT":                None,   # WT should not appear here
    }
    return s.map(mapping)


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
    parser.add_argument("--mode", choices=["multiclass", "hotspot", "binary_ms"],
                        default="multiclass",
                        help="multiclass: fine-grained subtype; hotspot: hotspot vs "
                             "non-hotspot missense; binary_ms: Missense vs WT binary")
    args = parser.parse_args()
    proc = args.proc_dir
    plots = proc / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    expr, labels = load_ccle()

    if args.mode == "binary_ms":
        # Missense vs WT — exclude all other mutant subtypes
        mask = labels["tp53_class"].isin(["Missense", "WT"])
        expr = expr[mask]
        y = labels.loc[mask, "tp53_class"]
        active_classes = ["WT", "Missense"]
        print(f"Missense-vs-WT mode: {mask.sum()} samples "
              f"({(y=='Missense').sum()} Missense, {(y=='WT').sum()} WT)")
    elif args.mode == "hotspot":
        # Hotspot vs non-hotspot within missense only
        mask = labels["tp53_class"] == "Missense"
        expr = expr[mask]
        y = labels.loc[mask, "tp53_hotspot"].map({1: "Hotspot", 0: "Non-hotspot"})
        active_classes = ["Hotspot", "Non-hotspot"]
        print(f"Hotspot mode: {mask.sum()} missense samples "
              f"({(y=='Hotspot').sum()} hotspot, {(y=='Non-hotspot').sum()} non-hotspot)")
    else:
        # Default: fine-grained multiclass on mutant subset
        is_mut = labels["tp53_binary"] == 1
        expr = expr[is_mut]
        y_raw = labels.loc[is_mut, "tp53_class"]
        y_mapped = merge_classes(y_raw)
        # Drop classes that are None (WT / Other / unmapped) or too small
        y_mapped = y_mapped.dropna()
        expr = expr.loc[y_mapped.index]
        # Drop classes below minimum sample size
        counts = y_mapped.value_counts()
        valid_classes = counts[counts >= MIN_CLASS_SIZE].index.tolist()
        # Preserve biologically meaningful order
        active_classes = [c for c in CLASSES if c in valid_classes]
        keep = y_mapped.isin(active_classes)
        y = y_mapped[keep]
        expr = expr.loc[y.index]
        dropped = counts[~counts.index.isin(valid_classes)]
        if not dropped.empty:
            print(f"Dropped under-represented classes (< {MIN_CLASS_SIZE} samples): "
                  f"{dropped.to_dict()}")

    print(f"TP53-mutant cell lines: {len(y)}")
    dist = y.value_counts().reindex(active_classes, fill_value=0)
    print("\nClass distribution:")
    print(dist.to_string())
    dist_df = dist.rename_axis("class").reset_index(name="count")
    dist_df["fraction"] = dist_df["count"] / dist_df["count"].sum()
    dist_df.to_csv(proc / f"multiclass_class_distribution_{args.mode}.csv", index=False)

    CLASSES_ACTIVE = active_classes

    top_genes_full = pd.read_csv(proc / "top_genes.csv")["gene"].tolist()
    # Align gene columns: expression matrix may use different column name style
    avail = [g for g in top_genes_full if g in expr.columns]
    X = expr[avail].values.astype(np.float32)
    print(f"\nFeatures: {X.shape}  (top-2k HVG from binary task, {len(avail)} matched)")

    le = {c: i for i, c in enumerate(CLASSES_ACTIVE)}
    y_int = y.map(le).values

    n_cls = len(CLASSES_ACTIVE)
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    oof_proba = {m: np.zeros((len(y), n_cls)) for m in ["xgb", "logreg"]}
    fold_metrics: dict[str, list[dict]] = {m: [] for m in oof_proba}
    importance_per_class = np.zeros((n_cls, X.shape[1]))

    xgb_obj = "binary:logistic" if n_cls == 2 else "multi:softprob"

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_int)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_int[train_idx], y_int[test_idx]

        # ── XGBoost ──────────────────────────────────────────────────────
        xgb_params = dict(
            n_estimators=400, max_depth=4, learning_rate=0.03,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method="hist", random_state=args.seed, n_jobs=-1,
        )
        if n_cls == 2:
            xgb_clf = xgb.XGBClassifier(objective=xgb_obj, **xgb_params)
        else:
            xgb_clf = xgb.XGBClassifier(
                objective=xgb_obj, num_class=n_cls, **xgb_params,
            )
        xgb_clf.fit(X_train, y_train)
        proba_raw = xgb_clf.predict_proba(X_test)
        if n_cls == 2 and proba_raw.ndim == 1:
            proba_raw = np.column_stack([1 - proba_raw, proba_raw])
        oof_proba["xgb"][test_idx] = proba_raw

        # Per-class importance via one-vs-rest XGB on the train fold
        for c in range(n_cls):
            y_bin = (y_train == c).astype(int)
            if y_bin.sum() < 5:
                continue
            ovr = xgb.XGBClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                tree_method="hist", random_state=args.seed, n_jobs=-1,
                objective="binary:logistic",
            )
            ovr.fit(X_train, y_bin)
            importance_per_class[c] += ovr.feature_importances_

        # ── Logistic Regression ───────────────────────────────────────────
        scaler = StandardScaler()
        Xs_train = scaler.fit_transform(X_train)
        Xs_test = scaler.transform(X_test)
        lr_clf = LogisticRegression(
            penalty="l2", C=1.0, solver="lbfgs", max_iter=2000,
            n_jobs=-1, random_state=args.seed,
        )
        lr_clf.fit(Xs_train, y_train)
        oof_proba["logreg"][test_idx] = lr_clf.predict_proba(Xs_test)

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
        "mode": args.mode,
        "classes": CLASSES_ACTIVE,
        "class_distribution": dist.to_dict(),
        "n_features": int(X.shape[1]),
        "n_splits": args.n_splits,
        "models": {},
    }
    per_class_rows = []
    for m in oof_proba:
        proba = oof_proba[m]
        y_pred = proba.argmax(axis=1)
        inv_le = {i: c for c, i in le.items()}
        y_true_lbl = pd.Series(y_int).map(inv_le).values
        y_pred_lbl = pd.Series(y_pred).map(inv_le).values

        agg = evaluate(y_true_lbl, y_pred_lbl, proba, CLASSES_ACTIVE)
        agg["per_fold"] = fold_metrics[m]
        summary["models"][m] = agg

        pcm = per_class_metrics(y_true_lbl, y_pred_lbl, CLASSES_ACTIVE)
        pcm["model"] = m
        per_class_rows.append(pcm)

        plot_confusion(y_true_lbl, y_pred_lbl, CLASSES_ACTIVE,
                       m.upper(), plots / f"multiclass_confusion_{args.mode}_{m}.png")

    pd.concat(per_class_rows, ignore_index=True).to_csv(
        proc / f"multiclass_per_class_metrics_{args.mode}.csv", index=False
    )

    # OOF predictions
    oof_df_parts = []
    for m, proba in oof_proba.items():
        df = pd.DataFrame(proba, columns=[f"{m}_{c}" for c in CLASSES_ACTIVE],
                          index=expr.index)
        oof_df_parts.append(df)
    oof_df = pd.concat([
        pd.DataFrame({"true_class": y.values}, index=expr.index),
        *oof_df_parts,
    ], axis=1)
    oof_df.index.name = "ModelID"
    oof_df.to_csv(proc / f"multiclass_oof_preds_{args.mode}.csv")

    # ── Per-class top-K genes (XGBoost OvR aggregated importance) ──────
    importance_per_class /= args.n_splits  # mean across folds
    top_rows = []
    for c, name in enumerate(CLASSES_ACTIVE):
        order = np.argsort(-importance_per_class[c])[:args.top_k_genes]
        for rank, idx in enumerate(order, start=1):
            gene_full = avail[idx]
            top_rows.append({
                "class": name, "rank": rank,
                "gene_full": gene_full,
                "gene_symbol": gene_full.split(" (")[0],
                "mean_xgb_importance": float(importance_per_class[c, idx]),
            })
    top_df = pd.DataFrame(top_rows)
    top_df.to_csv(proc / f"top_genes_multiclass_{args.mode}.csv", index=False)

    with open(proc / f"multiclass_metrics_{args.mode}.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # ── Plot — per-class F1 bar ─────────────────────────────────────────
    pcm_xgb = per_class_rows[0]; pcm_lr = per_class_rows[1]
    fig, ax = plt.subplots(figsize=(max(7.5, len(CLASSES_ACTIVE) * 1.5), 5))
    width = 0.4
    x_pos = np.arange(len(CLASSES_ACTIVE))
    ax.bar(x_pos - width / 2, pcm_xgb["f1"], width, color="steelblue",
           edgecolor="white", label="XGBoost")
    ax.bar(x_pos + width / 2, pcm_lr["f1"], width, color="tomato",
           edgecolor="white", label="LogReg")
    for i, (a, b, support) in enumerate(zip(pcm_xgb["f1"], pcm_lr["f1"],
                                             pcm_xgb["support"])):
        ax.text(i - width / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + width / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.text(i, -0.05, f"n={support}", ha="center", fontsize=8, color="gray")
    ax.set_xticks(x_pos); ax.set_xticklabels(CLASSES_ACTIVE)
    ax.set_ylim(-0.08, 1.0); ax.set_ylabel("Per-class F1 (OOF)")
    ax.set_title(f"Task 2 [{args.mode}] — Per-class F1, TP53 mutation-type classification")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots / f"multiclass_per_class_f1_{args.mode}.png", dpi=150)
    plt.close()

    # ── Print final summary ────────────────────────────────────────────
    print("\n=== Multiclass OOF results (n=%d, %d classes, mode=%s) ===" % (
        len(y), len(CLASSES_ACTIVE), args.mode))
    for m in oof_proba:
        a = summary["models"][m]
        print(f"  {m:>6s}:  acc={a['accuracy']:.4f}  macroF1={a['macro_f1']:.4f}  "
              f"weightedF1={a['weighted_f1']:.4f}  AUC_OvR={a['roc_auc_ovr_macro']:.4f}")
    print("\nPer-class metrics:")
    print(pd.concat(per_class_rows, ignore_index=True).round(4).to_string(index=False))

    print(f"\nTop-{args.top_k_genes} genes per class saved → "
          f"{proc / f'top_genes_multiclass_{args.mode}.csv'}")
    print(f"All artefacts → {proc}")


if __name__ == "__main__":
    main()
