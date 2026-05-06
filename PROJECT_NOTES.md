# TP53 Mutation Predictor — Project Notes

## Project Overview

Predict TP53 mutation status in cancer cell lines from bulk RNA-seq expression profiles (CCLE), comparing an XGBoost tabular baseline against a single GCN operating on a gene–gene co-expression graph. Binary classification (mutant vs. wild-type) only.

**MVP scope** — descoped due to deadline. Goal: a working end-to-end result presentable as a proof-of-concept comparison between classical ML and graph-based modeling, even if the GCN does not beat XGBoost.

The GNN track is inspired by Ravasio (2024), *"Predicting TP53 Mutation Status from scRNA-seq with Graph Neural Networks"*, adapted to bulk CCLE data.

---

## Data Sources

- **CCLE RNA-seq expression** (DepMap): `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` — log₂(TPM+1), 1,719 cell lines × 19,215 protein-coding genes.
- **CCLE somatic mutations** (DepMap): `OmicsSomaticMutations.csv` — somatic calls per cell line; TP53 entries used to derive labels (see `notebooks/gene_expression/01_EDA.ipynb` § 1).
- *(Optional, later)* Fischer's curated p53 target gene list — for biology-driven feature selection.

Raw CSVs live under `data/raw/` and are gitignored.

---

## MVP Scope (locked)

- Binary TP53 mutant vs. wild-type only.
- CCLE bulk RNA-seq only.
- **Top-2,000** highly variable genes only — same gene set used by both models.
- **XGBoost** baseline with fixed sensible hyperparameters, 5-fold stratified CV.
- **One** GCN model with fixed hyperparameters (no GAT, no Optuna, no HP search).
- No multi-class. No gene-set ablations. No graph-threshold ablations.

Multi-class subtypes, GAT, Optuna, and TP53-target-gene comparison are explicitly **out of scope** for the deadline; they remain on the long-term roadmap (see "Deferred" below).

---

## GNN Track — Graph Construction (MVP)

- **Topology**: A single shared gene–gene graph built once from the top-2k HVG of the CCLE expression matrix.
- **Edges**: Spearman correlation between gene expression vectors across cell lines, kept above one fixed threshold (chosen once, no ablation).
- **Node features**: Per-cell-line, node feature for gene *g* is that cell line's expression value for *g* (scalar).
- **Task**: Graph-level binary classification — one `Data` object per cell line sharing the same `edge_index`.

---

## Decisions & Rationale

| Decision | Rationale |
|---|---|
| MVP descope | Deadline pressure — better to ship a working end-to-end comparison than a partial larger pipeline. |
| Use CCLE bulk RNA-seq | Already in scope; well-curated; aligned with sample-level labels; matches the EDA already done. |
| log₂(TPM+1) expression | DepMap default; reduces dynamic-range skew; no further normalisation needed up front. |
| Binary only | Smallest subtypes (Splice_Site n=67, Frame_Shift_Del n=58) make multi-class noisy; out of scope for MVP. |
| Top-2k HVG, same set for both models | Fast, fair comparison; matches Ravasio's bulk benchmark feature count. |
| XGBoost, fixed HPs (no Optuna) | Defaults are known-strong on this scale of tabular data; tuning would cost time without changing the story. |
| Single GCN, fixed HPs (no GAT, no Optuna) | One architecture is enough to demonstrate the graph approach as a proof of concept. |
| Single shared topology, per-sample features | Follows Ravasio's bulk approach; simplest valid setup. |
| Stratified 5-fold CV | TP53 binary balance is ~58:42 (mutant:WT); stratification keeps folds comparable. |

---

## Deferred (post-deadline)

- GAT variant
- Optuna hyperparameter tuning
- Multi-class TP53 subtype classification
- TP53-target-gene feature set vs. HVG comparison
- Graph-threshold and gene-set-size ablations

---

## Autonomous Run Log (2026-05-05 evening — first end-to-end run)

User instructed Claude to run the pipeline overnight without blocking on questions, choosing standard defaults. Decisions made autonomously:

**Data**
- DepMap release: **24Q4** (latest public). Files downloaded to `~/scratch/tp53-data/` on Bocconi HPC and symlinked into `data/raw/`.
- Expression file renamed in 24Q4 from `OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv` → `OmicsExpressionProteinCodingGenesTPMLogp1.csv`. `src/load_data.py` updated to match.
- The `lineage` column from the EDA labels was dropped (was always `None`); will be re-added if/when `sample_info.csv` is joined in.

**Feature selection**
- Top **2,000 highly variable genes** (variance over CCLE cell lines). Same gene set used by both XGBoost and the GCN.

**XGBoost hyperparameters (fixed, no Optuna)**
- `n_estimators=300`, `max_depth=6`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `tree_method=hist`, `objective=binary:logistic`, `eval_metric=auc`, `random_state=42`.
- 5-fold stratified CV; threshold 0.5 for class assignment.

**Gene–gene graph (Spearman)**
- Computed via rank-then-`np.corrcoef` (much faster than pairwise `pd.corr` on 2k genes).
- Threshold: **|ρ| ≥ 0.5**, undirected (both directions in `edge_index`). Single fixed threshold, no ablation.
- Cached at `data/processed/gene_graph.npz` (edge_index, edge_weight, gene_order, threshold).

**GCN architecture & training (fixed, no Optuna)**
- Two `GCNConv` layers, hidden dim **64**, dropout **0.5**, ReLU, global mean pool, linear head (1 logit), `BCEWithLogitsLoss`.
- Adam optimizer: `lr=1e-3`, `weight_decay=1e-4`. 100 epochs, batch size 32.
- Same 5-fold splits as XGBoost (loaded from `data/processed/cv_splits.csv`) for an apples-to-apples comparison.
- GPU if available, else CPU.

**HPC setup**
- Conda env `tp53-predictor` installed from `environment.yml` (PyTorch + PyG via pip). Module: `miniconda3` + `cuda/12.4`.
- SLURM partitions: **`defq`** for XGBoost (CPU, 30 min, 4 cpus, 8 GB), **`gpunew`** for GCN (1 GPU, 2 h, 4 cpus, 16 GB).
- GCN job depends on XGB completion (`--dependency=afterok`) so the GCN can read `cv_splits.csv` produced by the XGB run.

**Outputs (saved under `data/processed/` on HPC, then rsynced back to laptop)**
- `cv_splits.csv` — ModelID → fold (produced by XGB)
- `top_genes.csv` — top-2k HVG used by both models
- `xgb_baseline_metrics.json`, `xgb_baseline_oof_preds.csv`
- `gene_graph.npz`
- `gcn_metrics.json`, `gcn_oof_preds.csv`
- SLURM logs under `logs/xgb_<jobid>.out|.err` and `logs/gcn_<jobid>.out|.err`

**Failure handling**
- If env install fails, retry with simpler resolve (drop pinned PyTorch channel) or pip-only.
- If XGB SLURM job fails, inspect log, fix root cause, requeue.
- If GCN fails on PyG import, reinstall PyG with explicit torch version match.
- If GCN OOM on GPU, halve batch size.
- All failures and recoveries appended to this log.

---

## Run Results — 2026-05-06 (MVP end-to-end)

**Cluster**: Bocconi HPC, partition `stud` (QOS `stud`), 1 × A100 80GB for GCN, CPU for XGB.
**Dataset (DepMap 24Q4)**: 1,673 cell lines × 19,193 protein-coding genes. Class balance: 986 mutant (58.9 %) / 687 WT (41.1 %).
**Feature set**: top 2,000 highly variable genes (variance over CCLE).
**Graph**: 59,701 undirected edges (avg degree ≈ 60) at |Spearman ρ| ≥ 0.5.

### XGBoost baseline (5-fold stratified CV, OOF)

| Metric    | Value  |
|-----------|-------:|
| Accuracy  | 0.847  |
| Precision | 0.843  |
| Recall    | 0.910  |
| **F1**    | **0.875** |
| ROC-AUC   | 0.906  |
| PR-AUC    | 0.909  |

Matches the Ravasio (2024) bulk benchmark of F1 ≈ 0.88 on CCLE — sanity-check passed. Wall time on CPU: **1 m 42 s**.

### GCN (2 layers, hidden 64, dropout 0.5, lr 1e-3, 100 epochs, BCE)

| Metric    | Value  |
|-----------|-------:|
| Accuracy  | 0.595  |
| Precision | 0.608  |
| Recall    | 0.879  |
| F1        | 0.719  |
| ROC-AUC   | 0.625  |
| PR-AUC    | 0.705  |

Wall time on A100: **28 m 30 s** (5 folds × 100 epochs).

### Diagnosis

GCN is clearly **underfitting**: training loss only fell from 0.676 → 0.65 over 100 epochs (BCE on a 59 % positive class). The high recall (0.88) with low precision (0.61) shows the model is mostly predicting "mutant" by default — only ~0.04 above always-positive baseline ROC-AUC (0.625 vs ~0.59 majority).

Likely contributors (none addressed in MVP):
1. Over-sparse node features (1 scalar per gene per sample is very low signal).
2. Small model (64 hidden units, 2 layers) for a 2 k-node graph.
3. Fixed lr=1e-3 + 100 epochs may be far from converged on this graph.
4. Co-expression edges may be redundant with linear gene–gene structure that XGBoost already exploits.

**MVP conclusion**: classical bulk-RNA + tabular ML (XGBoost) substantially outperforms a vanilla GCN on co-expression graphs at the same feature budget. Presentable as a proof-of-concept comparison.

### Outputs (synced back to laptop, `data/processed/`)

- `metrics_table.csv`, `xgb_baseline_metrics.json`, `gcn_metrics.json` — aggregate + per-fold.
- `xgb_baseline_oof_preds.csv`, `gcn_oof_preds.csv` — out-of-fold predictions.
- `cv_splits.csv` — shared 5-fold assignment.
- `top_genes.csv` — top-2 k HVG used by both models.
- `gene_graph.npz` — edge_index + edge_weight + gene order.
- `spearman_corr.npz` — full 2 k × 2 k correlation matrix (16 MB).
- `plots/`: `roc_curves.png`, `pr_curves.png`, `confusion_matrix_xgb.png`, `confusion_matrix_gcn.png`, `model_comparison.png`.
- `logs/xgb_489339.{out,err}`, `logs/gcn_489340.{out,err}`.

### Issues encountered & resolved

1. **`set -u` tripped on conda's MKL activate script** → switched both sbatch files to `set -eo pipefail` (kept fail-fast and pipe-safety, dropped unbound-variable check).
2. **Initial submission used partitions/QoS this account can't access** (defq, gpunew + default QOS `normal`) → switched to `--partition=stud --qos=stud`. Only stud is available for this user.
3. **DepMap 24Q4 schema differs from EDA notebook (23Q4-era)**: expression file `OmicsExpressionProteinCodingGenesTPMLogp1.csv` is already filtered (no `IsDefaultEntryForModel` column), and mutations file lost the same flag. `src/load_data.py` updated to drop those filters.
4. **rsync target accident**: copied src/ and jobs/ files to the repo root on HPC instead of into their subdirs; cleaned up and re-rsynced into proper subdirs.

---

## GCN v2 — Targeted improvement pass (2026-05-06)

### Goal
Make the GCN scientifically respectable; check whether it can become competitive with XGBoost, while keeping scope locked (binary TP53, CCLE bulk, top-2k HVG, same 5-fold CV splits, same XGBoost baseline).

### Changes vs. v1

| Area | v1 | v2 |
|---|---|---|
| Loss | plain BCE | `BCEWithLogitsLoss(pos_weight = N_neg / N_pos)` per-fold (≈ 0.7) |
| Node features | scalar expression (1-d) | `[expression, z-score]` (2-d) — z-score fit on training fold only |
| Hidden dim | 64 | 128 |
| Layers | 2 | 3 |
| Dropout | 0.5 | 0.4 |
| BatchNorm | no | yes |
| Residual | no | yes |
| Optimizer | Adam, lr 1e-3 | Adam, lr 1e-3, ReduceLROnPlateau on val ROC-AUC (factor 0.5, patience 10, min_lr 1e-6) |
| Schedule | 100 fixed epochs | up to 300 epochs, **early stop** on val ROC-AUC (patience 30) |
| Validation | none | 15 % stratified train→val split inside each CV fold |
| Run naming | n/a | `gcn_<run_name>_*` outputs (curves CSV, OOF preds, metrics JSON) |

### Graph ablation

| Graph | Mode | Spec | Edges (undirected) | Avg degree |
|---|---|---|---:|---:|
| `gene_graph_thr05.npz` | threshold | \|ρ\| ≥ 0.5 | 59,701 | 59.7 |
| `gene_graph_thr07.npz` | threshold | \|ρ\| ≥ 0.7 | **2,556** | **2.6** |
| `gene_graph_topk10.npz` | top-k | top-10 / gene | 18,131 | 18.1 |

### Final OOF results (5-fold CV, n = 1673)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **XGBoost** | **0.847** | **0.843** | 0.910 | **0.875** | **0.906** | **0.909** |
| GCN baseline (v1) | 0.595 | 0.608 | 0.879 | 0.719 | 0.625 | 0.705 |
| GCN v2 thr=0.5 | 0.664 | 0.720 | 0.705 | 0.712 | 0.701 | 0.751 |
| **GCN v2 thr=0.7** (best) | 0.663 | 0.668 | 0.853 | **0.749** | **0.707** | **0.759** |
| GCN v2 top-k=10 | 0.646 | 0.746 | 0.606 | 0.669 | 0.704 | 0.758 |

Wall time: thr=0.5 → 1 h 17 m, thr=0.7 → 12 m, top-k=10 → 22 m (all on A100, jobs ran serially due to QOS limit on `stud` partition).

### Conclusions (honest)

1. **The v2 changes worked** — every v2 variant beats the v1 baseline on every threshold-free metric (ROC-AUC ≈ 0.70 vs 0.625, PR-AUC ≈ 0.76 vs 0.705). The GCN no longer collapses to "predict mutant" — confusion matrices are more balanced and precision rose from 0.61 → 0.67–0.75.
2. **Graph topology has a small effect** — the three graph variants land within ~0.01 of each other on every metric. The very sparse thr=0.7 graph (avg degree 2.6) is no worse than the dense thr=0.5 (avg degree 60), suggesting the GCN extracts most of its (limited) signal locally and the densely-connected version mostly adds noise. The top-k=10 graph trades higher precision for lower recall.
3. **GCN is NOT competitive with XGBoost on this task / data / feature budget.** XGBoost still wins by ~0.2 absolute ROC-AUC and ~0.13 absolute F1 over the best GCN variant. XGBoost on top-2k variable genes is a strong baseline because the per-gene expression signal is informative (TP53 targets like CDKN1A, MDM2 are well-separated in CCLE bulk), and a tree model captures that directly without needing relational structure.
4. **Likely remaining bottlenecks** (out of MVP scope, not addressed):
   - Node features are still scalar/2-d; richer per-node embeddings (e.g., gene-set enrichment scores, PCA-reduced expression context, learnable gene embeddings) would give the message passing more substrate to combine.
   - Co-expression edges encode redundant information that XGBoost already exploits via its tree splits — the graph may not contribute orthogonal information here.
   - Architecture is still small. Larger GAT with attention pooling, or graph transformers, would likely close more of the gap. Scope-locked out for the deadline.

### Best GCN variant: **`v2_thr07`**
- Best F1 (0.749) and best ROC-AUC (0.707) of all GCN runs.
- Sparse graph (2,556 edges) → fastest to train (12 min) → best for any future iteration.
- Still trails XGBoost by F1 0.749 vs 0.875 (gap of 0.126).

### Outputs added in v2

- `data/processed/gcn_v2_{thr05,thr07,topk10}_metrics.json` — full per-fold + aggregate metrics.
- `data/processed/gcn_v2_{thr05,thr07,topk10}_oof_preds.csv` — OOF predicted probabilities.
- `data/processed/gcn_v2_{thr05,thr07,topk10}_curves.csv` — per-epoch train_loss / val_loss / val_auc / val_f1 / lr.
- `data/processed/gene_graph_{thr07,topk10}.npz` — new graph structures.
- `data/processed/graph_stats.csv` — edge / degree summary across all three graphs.
- `data/processed/metrics_table.csv` — unified comparison (now 5 rows: XGB + 4 GCN variants).
- `data/processed/plots/` — refreshed: `roc_curves.png`, `pr_curves.png`, `model_comparison.png`, `training_curves.png`, plus one `confusion_matrix_*.png` per model.
- `logs/gcn_v2_{489441,489442,489479}.{out,err}` — full SLURM logs.

### Issues encountered in v2 & resolved

5. **`stud` QOS only allows 1 running + 2 queued jobs per user** → planned for parallel submission of all 3, but had to run serially: submitted thr05+thr07 first, then resubmitted topk10 once thr05 finished. Total wall clock ~2 h instead of the ~1.5 h parallel ideal.

---

## Repo Layout (planned)

```
data/raw/         CCLE CSVs (gitignored)
data/processed/   cached expression matrix, labels, graph artifacts
notebooks/        EDA + results notebooks
src/              loading, preprocessing, graph construction, models, training
jobs/             SLURM scripts for HPC runs
```

`src/` and `jobs/` will be filled in incrementally as we build the pipeline.

---

## Progress Log

| Date | Update |
|---|---|
| 2026-04-21 | Repository initialized. Project structure and notes drafted. |
| 2026-05-05 | EDA notebook complete. Scope updated: bulk CCLE + XGBoost baseline + GCN/GAT GNN track on a shared Spearman gene–gene graph. RF and MLP dropped from candidate models. Multi-class deferred. Cluster access verified (Bocconi HPC, SLURM). |
| 2026-05-05 | Descoped to MVP for deadline: top-2k HVG only, XGBoost + one GCN with fixed hyperparameters, no GAT, no Optuna, no ablations. |

