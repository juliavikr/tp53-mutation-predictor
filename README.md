# TP53 Mutation Predictor

A biologically informed machine learning pipeline for predicting TP53 mutation status and subtype from bulk RNA-seq gene expression, and for characterising how TP53 mutations reshape the transcriptome.

> An MSc-level computational oncology project — built on CCLE (DepMap 24Q4) cell-line transcriptomes, validated on 8,424 TCGA primary tumours, and interpreted against a curated TP53 pathway gene set.

---

## TL;DR

- **Task** — binary + multi-class classification: TP53 mutant vs wild-type (and subtype) from gene expression; plus differential expression to identify genes altered by TP53 mutation.
- **Best model** — XGBoost on top-2,000 HVGs: **F1 = 0.875, ROC-AUC = 0.906** on CCLE 5-fold CV; **F1 = 0.604, ROC-AUC = 0.806** on 8,424 TCGA primary tumours (z-scored per cohort).
- **Best GNN** — GAT on a hybrid graph (STRING PPI ∪ co-expression): **F1 = 0.760, ROC-AUC = 0.706** on CCLE — best GNN variant but still behind XGBoost.
- **Top SHAP feature** — **CDKN1A (p21)** dominates with mean |SHAP| = 1.27, 3× the next gene. Seven of the top 20 features are direct TP53 transcriptional targets — the model independently rediscovered the canonical TP53 → p21 axis.
- **Key finding** — GNN models do not transfer to TCGA under per-cohort z-score normalisation (AUC ≈ 0.4), while XGBoost transfers to AUC = 0.806. Likely cause: BatchNorm distribution shift between CCLE (log₂ TPM+1) and TCGA (log₂ norm_count+1).

---

## Research questions

1. **Expression → Mutation (prediction):** Can we predict TP53 mutation status (binary) and subtype (multi-class) from bulk RNA-seq profiles?
2. **Mutation → Expression (effect):** Which genes are differentially expressed as a consequence of TP53 mutation, and what biological processes do they represent?

Both questions are addressed on two independent datasets (CCLE cell lines and TCGA primary tumours) using a consistent, modular pipeline.

---

## Datasets

| Dataset | Source | Samples | Expression format | Labels |
|---------|--------|---------|-------------------|--------|
| **CCLE** | DepMap portal (24Q4) | 1,673 cell lines | log₂(TPM+1) | `tp53_binary`, `tp53_class` (5-way) |
| **TCGA** | UCSC Xena PanCancer Atlas | 8,424 primary tumours | log₂(norm_count+1) | `tp53_binary`, `cancer_type` |
| **TCGA (full)** | GDC RSEM + MC3 MAF | 9,875 primary tumours | log₂(RSEM+1) | `tp53_binary`, `tp53_class` (7-way) |

TCGA serves as a **held-out external validation set** for models trained on CCLE — no TCGA samples are used during training.

---

## Key results (binary classification, CCLE)

| Model | F1 | ROC-AUC | Notes |
|-------|----|---------|-------|
| XGBoost (top-2k HVG) | **0.875** | **0.906** | Matches Ravasio 2024 benchmark |
| **GAT, hybrid graph** | **0.760** | **0.706** | Best GNN — STRING PPI ∪ co-expression |
| GCN v2, Spearman thr=0.7 | 0.749 | 0.707 | Sparse co-expression graph |
| GCN v2, Spearman thr=0.5 | 0.712 | 0.701 | |
| GCN v2, top-k=10 | 0.669 | 0.704 | |
| GCN v2, STRING PPI | 0.668 | 0.678 | Biology-only graph |
| GAT, Spearman thr=0.7 | 0.680 | 0.622 | |

**TCGA external validation (XGBoost, CCLE-trained):** ROC-AUC = 0.806, F1 = 0.604.
Domain shift: CCLE has 58.9% TP53 mutation rate vs TCGA 36.5%. GNNs do not transfer (AUC ≈ 0.4 on TCGA).

**Top predictive gene (SHAP):** CDKN1A (p21) — mean |SHAP| = 1.27, far above all others. 7 of the top 20 SHAP genes are confirmed TP53 pathway members.

---

## Repository layout

```
tp53-mutation-predictor/
├── data/
│   ├── raw/                        # Downloaded source files (gitignored)
│   │   ├── OmicsExpression...csv   # CCLE expression (DepMap 24Q4)
│   │   ├── OmicsSomaticMutations.csv
│   │   └── tcga/                   # TCGA Xena files
│   └── processed/                  # Generated artefacts
│       ├── top_genes.csv           # Top-2k CCLE HVGs
│       ├── cv_splits.csv           # 5-fold stratified split assignments
│       ├── tcga_expression.csv     # TCGA aligned to CCLE HVGs
│       ├── tcga_labels.csv         # TCGA binary labels + cancer type
│       ├── tcga_preprocessed.csv.gz # TCGA full (9,875 × 18,253) with multi-class labels
│       ├── gene_graph_*.npz        # Gene co-expression / PPI graphs
│       ├── *_metrics.json          # Per-model evaluation metrics
│       ├── shap_top20.csv          # SHAP feature importance
│       └── plots/                  # All figures
│
├── notebooks/
│   ├── gene_expression/
│   │   └── 01_EDA.ipynb            # CCLE exploratory data analysis
│   ├── 02_tcga_data_loading.ipynb  # TCGA loading + merging (GDC RSEM + MC3 MAF)
│   ├── 03_tcga_eda.ipynb           # TCGA EDA
│   └── 04_tcga_preprocessing.ipynb # TCGA preprocessing → tcga_preprocessed.csv.gz
│
├── src/
│   ├── load_data.py                # Load CCLE expression + derive labels
│   ├── train_xgb.py                # XGBoost 5-fold CV
│   ├── train_gnn.py                # GCN / GAT training
│   ├── gcn.py                      # GCN architecture
│   ├── gat.py                      # GAT architecture
│   ├── graph_construction.py       # Spearman co-expression graphs
│   ├── build_bio_graph.py          # STRING PPI graph
│   ├── shap_analysis.py            # SHAP feature importance
│   ├── tcga_load.py                # Prepare TCGA for modeling
│   ├── tcga_eval.py                # XGBoost CCLE→TCGA transfer
│   ├── tcga_gnn_eval.py            # GNN CCLE→TCGA transfer
│   ├── domain_comparison.py        # CCLE vs TCGA distribution analysis
│   ├── make_plots.py               # Unified ROC / PR / CM plots
│   └── tp53_pathway.py             # TP53 pathway gene annotations
│
├── jobs/                           # SLURM job scripts (Bocconi HPC)
├── logs/                           # SLURM output logs
├── PROJECT_NOTES.md                # Detailed run log and decisions
└── environment.yml                 # Conda environment
```

---

## Setup

### Local
```bash
conda env create -f environment.yml
conda activate tp53-predictor
```

### HPC (Bocconi cluster, SLURM)
```bash
module load miniconda3 cuda/12.4
conda env create -f environment.yml
conda activate tp53-predictor
```

---

## Data download

### CCLE (DepMap 24Q4)
1. Go to [depmap.org/portal/data_page](https://depmap.org/portal/data_page/)
2. Select release **24Q4**
3. Download and place in `data/raw/`:
   - `OmicsExpressionProteinCodingGenesTPMLogp1.csv`
   - `OmicsSomaticMutations.csv`

### TCGA — Xena path (binary labels, default)
Files are downloaded automatically by `src/tcga_load.py` on first run:
```bash
python src/tcga_load.py
```

### TCGA — GDC path (multi-class labels, notebooks 02–04)
1. Download manually and place in `data/raw/`:
   - `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv` (~1.8 GB)
   - `mc3.v0.2.8.PUBLIC.maf` (~2.5 GB uncompressed)
2. Run notebooks `02 → 03 → 04` in order
3. Output: `data/processed/tcga_preprocessed.csv.gz`

---

## Reproducing results

### Step 1 — CCLE feature selection + XGBoost baseline
```bash
python src/train_xgb.py
# Outputs: data/processed/top_genes.csv, cv_splits.csv, xgb_baseline_metrics.json
```

### Step 2 — Build gene co-expression graphs
```bash
python src/graph_construction.py --mode threshold --threshold 0.5
python src/graph_construction.py --mode threshold --threshold 0.7
python src/graph_construction.py --mode topk --top-k 10
python src/build_bio_graph.py
```

### Step 3 — Train GCN / GAT (HPC recommended)
```bash
sbatch jobs/train_gnn_v2.sbatch   # GCN variants
sbatch jobs/train_gat.sbatch      # GAT
```

### Step 4 — SHAP analysis
```bash
python src/shap_analysis.py
```

### Step 5 — TCGA external validation
```bash
# Binary (Xena path, already run)
python src/tcga_eval.py

# Multi-class (notebook path)
python src/tcga_load.py --from-preprocessed
python src/tcga_eval.py
```

### Step 6 — Generate all plots
```bash
python src/make_plots.py
```

---

## Pipeline diagram

```
CCLE raw files
  └─ load_data.py ──────────────────────────────────┐
      ├─ train_xgb.py          XGB F1=0.875          │
      ├─ graph_construction.py  5 graph variants      │  CCLE model
      │   └─ train_gnn.py      GCN F1=0.749          │
      └─ shap_analysis.py      CDKN1A top feature     │
                                                      │
TCGA Xena files                                       │ transfer
  └─ tcga_load.py ──────────────────────────────────►├─ tcga_eval.py     AUC=0.806
                                                      └─ tcga_gnn_eval.py (pending)

TCGA GDC files (notebooks 02-04)
  └─ tcga_preprocessed.csv.gz  (multi-class labels, 18k genes)
      ├─ tcga_load.py --from-preprocessed
      └─ [multi-class modeling — in progress]
```

---

## Planned next steps

- [ ] Multi-class classification (mutation subtype) on CCLE and TCGA
- [ ] Differential expression analysis (mutation → transcriptome effects)
- [ ] Feature set comparison: all genes vs HVG vs TP53 targets vs DEGs
- [ ] GAT CCLE results + TCGA transfer
- [ ] Pathway enrichment on SHAP top-50 and DEG list
- [ ] Biological interpretation synthesis notebook

See `PROJECT_NOTES.md` for detailed decisions and run logs.
