# TP53 Mutation Predictor — Project Notes

## Project Overview

This project aims to predict TP53 mutation status in cancer cell lines from gene expression profiles (bulk RNA-seq) using the Cancer Cell Line Encyclopedia (CCLE) dataset. The goal is to build interpretable machine learning models that identify transcriptomic signatures associated with TP53 mutation.

---

## Data Sources

- **CCLE RNA-seq expression data**: Broad Institute DepMap portal (https://depmap.org/portal/)
  - File: `CCLE_expression.csv` — log2(TPM+1) gene expression matrix (cell lines × genes)
- **CCLE mutation data**: DepMap portal
  - File: `CCLE_mutations.csv` — somatic mutation calls per cell line
- **Sample info**: `sample_info.csv` — cell line metadata (lineage, tissue, etc.)

---

## Tasks

### Task 1: Binary Classification — Mutant vs. Wild-Type
Classify each cell line as TP53-mutant or TP53 wild-type based on its RNA-seq expression profile.

- **Input**: Gene expression vector per cell line
- **Output**: Binary label (mutant / WT)
- **Candidate models**: Logistic Regression, Random Forest, XGBoost, simple MLP
- **Evaluation**: AUC-ROC, F1, accuracy (stratified k-fold CV)

### Task 2: Mutation Type Classification
Multi-class classification of TP53 mutation subtypes (e.g., missense, nonsense, frameshift, splice site, WT).

- **Input**: Gene expression vector per cell line
- **Output**: Mutation type label
- **Candidate models**: Softmax Regression, Random Forest, XGBoost
- **Evaluation**: Macro F1, confusion matrix, AUC (one-vs-rest)

---

## Decisions & Rationale

| Decision | Rationale |
|---|---|
| Use CCLE as primary dataset | Large, well-curated, publicly available; RNA-seq + mutation data aligned per cell line |
| log2(TPM+1) expression values | Standard normalization; reduces dynamic range skew |
| Start with binary classification | Simpler baseline before tackling multi-class |
| Stratified k-fold CV | Class imbalance expected (TP53 mutant ~50% in CCLE but varies by lineage) |

---

## Progress Log

| Date | Update |
|---|---|
| 2026-04-21 | Repository initialized. Project structure and notes drafted. |

