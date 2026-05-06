# TCGA Pipeline Summary — Notebooks 02, 03, 04

**Project:** TP53 Mutation Predictor  
**Dataset covered:** TCGA pan-cancer (second dataset; CCLE is covered in notebook 01)  
**Notebooks:** `02_tcga_data_loading`, `03_tcga_eda`, `04_tcga_preprocessing`  
**Output:** `data/processed/tcga_preprocessed.csv.gz` — model-ready feature matrix

---

## Project Context

The overarching goal is to predict TP53 mutation status from bulk RNA-seq gene expression. The project runs two parallel tasks:

| Task | Description |
|------|-------------|
| **Binary classification** | Predict whether TP53 is mutated (1) or wild-type (0) |
| **Multi-class classification** | Predict the specific mutation type (Missense, Nonsense, Frameshift, Splice, etc.) |

Both tasks are performed on two datasets. CCLE (Cancer Cell Line Encyclopedia) is the cleaner, primary dataset — cell line data from a single standardised pipeline. TCGA (The Cancer Genome Atlas) is the second, harder dataset: primary tumor samples from a multi-centre cohort, with more biological and technical noise. **Notebooks 02–04 build and clean the TCGA dataset end-to-end.**

---

## Why TCGA is Harder than CCLE

| Aspect | CCLE | TCGA |
|--------|------|------|
| Sample type | Cancer cell lines | Primary tumors |
| Batch effects | Minimal | Substantial (multi-centre) — partially corrected by EB++ |
| Expression scale | log₂(TPM+1), pre-normalised | Raw RSEM — must log-transform |
| Mutation calling | Single pipeline | MC3 multi-caller consensus (more conservative) |
| Class noise | Low | Higher — tumor purity variation, stromal contamination |
| Sample size | ~1,775 cell lines | ~9,875 patients |

---

## Notebook 02 — TCGA Data Loading

### Objective
Load, parse, and merge two large raw files (expression + mutations) into a single patient-level feature matrix with TP53 labels.

### Data Sources

| File | Size | Description |
|------|------|-------------|
| `EBPlusPlusAdjustPANCAN_IlluminaHiSeq_RNASeqV2-v2.geneExp.tsv` | ~1.8 GB | TCGA PANCAN RNA-seq — EB++ batch-corrected RSEM values; genes × samples |
| `mc3.v0.2.8.PUBLIC.maf` | ~2.5 GB | MC3 pan-cancer somatic mutation calls; one row per mutation event |

Both files confirmed present before loading.

---

### Step 1 — Gene Expression Loading and Parsing

**Raw file shape:** 20,531 genes × 11,069 samples

Gene IDs were stored as `SYMBOL|ENTREZ_ID` format (e.g., `TP53|7157`). Parsing steps:
- Stripped the Entrez ID suffix, keeping gene symbol only
- Removed **29 genes** with `?` symbol (no gene annotation)
- Removed **1 duplicate** gene entry (multi-transcript genes)

**Genes retained: 20,501**

---

### Step 2 — Sample Filtering

After transposing to samples × genes format (11,069 × 20,501), samples were filtered to keep only **primary tumors**:

| Sample type code | Description | Count |
|-----------------|-------------|-------|
| `01` | Primary solid tumor | 9,706 |
| `03` | Primary blood-derived cancer | 173 |
| `11` | Solid tissue normal (excluded) | 737 |
| Other codes | Metastatic, recurrent, etc. (excluded) | 453 |

Normal tissue samples were excluded because we are predicting tumor-derived TP53 mutations.

After filtering: **9,879 samples → 9,875 unique patients** (4 duplicate barcodes removed by keeping the first sample per patient).

TCGA barcodes were truncated to the 12-character patient ID (`TCGA-TSS-participant`) to enable joining with the mutation table.

---

### Step 3 — MC3 Mutation Loading and TP53 Filtering

The full MC3 MAF file contains **3,600,963 mutation events** across **10,224 unique patients** (69 columns). Only 8 columns were loaded into memory to keep RAM usage manageable.

Filtering for TP53 with `FILTER == 'PASS'` (quality-controlled calls only):
- **3,729 TP53 PASS mutation events** across **3,325 unique patients**

Variant classification breakdown before deduplication:

| Classification | Count |
|---------------|-------|
| Missense_Mutation | 2,382 |
| Nonsense_Mutation | 488 |
| Frame_Shift_Del | 332 |
| Splice_Site | 238 |
| Frame_Shift_Ins | 101 |
| Silent | 90 |
| In_Frame_Del | 68 |
| In_Frame_Ins | 11 |
| Intron / 3'UTR / 5'UTR / Translation_Start_Site | 19 |

**Handling patients with multiple TP53 mutations:** some patients have compound heterozygous or multiple-hit mutations. One canonical mutation per patient was selected using a functional severity priority:

```
Nonsense > Frame_Shift_Del > Frame_Shift_Ins > Splice_Site > Missense > In_Frame_Del > In_Frame_Ins > ...
```

Post-deduplication: **3,325 unique patients** retained (no patient had their count reduced — each already had one dominant call).

---

### Step 4 — Merge

A **left join** was performed: all 9,875 expression patients are kept. Patients absent from the mutation table are labelled wild-type.

Two label columns were appended:

| Column | Type | Description |
|--------|------|-------------|
| `Mutated` | int (0/1) | 1 if the patient has a PASS-quality TP53 mutation |
| `Variant_Classification` | str | MAF mutation type string, or `WT` |

**Final merged shape: 9,875 patients × 20,503 columns** (20,501 gene features + 2 labels)

#### Binary label distribution

| Status | Count | Percentage |
|--------|-------|-----------|
| Wild-type (0) | 6,744 | 68.3% |
| Mutated (1) | 3,131 | 31.7% |

> **Class imbalance:** roughly 2:1 WT:mutant — moderate imbalance that must be addressed in modeling.

#### Multi-class label distribution (post-merge)

| Classification | Count |
|---------------|-------|
| WT | 6,744 |
| Missense_Mutation | 1,949 |
| Nonsense_Mutation | 452 |
| Frame_Shift_Del | 298 |
| Splice_Site | 215 |
| Frame_Shift_Ins | 96 |
| In_Frame_Del | 59 |
| Silent | 43 |
| In_Frame_Ins | 9 |
| Intron / 3'UTR / 5'UTR | 10 |

> **Severe imbalance in multi-class task:** Missense dominates (62.2% of all mutant samples). Rare classes like In_Frame_Ins (9 patients) will likely need to be collapsed or dropped.

#### Sanity checks

- TP53 expression column confirmed present (mean RSEM 1,638, range 26–15,568)
- No all-zero expression samples found
- **4,069,730 NaN values** detected in expression columns (flagged for EDA investigation)

**Output saved:** `data/processed/tcga_merged_raw.csv.gz` — 697.4 MB

---

## Notebook 03 — TCGA Exploratory Data Analysis

### Objective
Thoroughly characterise the raw merged dataset to identify data quality issues and motivate all preprocessing decisions made in notebook 04.

---

### 1. Missing Values — Structured Missingness

| Metric | Value |
|--------|-------|
| Total NaN cells | 4,069,730 |
| Gene columns with ≥1 NaN | 3,338 / 20,501 |
| Samples with ≥1 NaN | 1,690 / 9,875 |

**Key finding:** The missingness is highly structured — exactly **1,690 samples** are missing values for a large block of 3,338 specific genes, all with the same NaN count of 1,690. This is a systematic batch-level gap (certain genes not measured for a subset of TCGA cohorts), not random missingness. Filling with 0 (unexpressed) is a reasonable assumption for this pattern.

---

### 2. Class Balance

#### Binary target
- Mutation rate: **31.7%** (3,131 mutated / 9,875 total)
- Moderate imbalance requiring class weighting or SMOTE in modeling

#### Multi-class target (mutant patients only)
- **Missense fraction: 62.2%** (1,949 / 3,131)
- Severe long-tail distribution: In_Frame_Ins has only 9 samples
- Rare classes (Intron, 3'UTR, 5'UTR, In_Frame_Ins) are too small for reliable classification and will need to be grouped or dropped

---

### 3. Expression Distribution

Raw RSEM values are extremely right-skewed:

| Statistic | Value |
|-----------|-------|
| Mean | ~1,001 |
| Median | ~200 |
| 75th percentile | ~845 |
| Max | ~3,298,360 |
| Min | ~-0.99 |

> Note: slight negative values exist (a known artefact of the EB++ batch correction algorithm — not a biological signal).

After log₂(x+1) transformation, the distribution becomes bimodal: a sharp peak near zero (unexpressed genes) and a broader peak around log₂ ~6–10 (expressed genes). This is the expected shape for bulk RNA-seq data and confirms that log-transformation is necessary before modeling.

---

### 4. TP53 Expression by Mutation Status

TP53 mRNA level was examined as a solo predictor. The analysis confirms that **TP53 expression alone is not a reliable classifier**: many missense mutations are gain-of-function (the mutant protein accumulates, causing higher mRNA), while wild-type TP53 is tightly regulated at low basal levels. The boxplot shows overlapping distributions — reinforcing the need for genome-wide expression as features.

---

### 5. Per-Gene Quality Assessment

**Zero/near-zero expression genes:**
- Genes with mean RSEM == 0: a subset that carry no signal and should be dropped
- Genes with mean RSEM < 1.0: a broader set of very lowly expressed genes

**Coefficient of Variation (CV) analysis:**
- CV was computed on log₂-transformed values per gene
- 10th percentile CV threshold identified as the filter cut-off
- The CV distribution has a wide range, confirming that a large fraction of genes are nearly constant across patients and are unlikely to be informative predictors

---

### 6. PCA Sanity Check

PCA was run on 1,000 randomly selected genes (log₂-transformed, scaled):

| Component | Variance Explained |
|-----------|--------------------|
| PC1 | 10.19% |
| PC2 | 7.27% |
| PC3 | 5.16% |
| **Cumulative (3 PCs)** | **22.62%** |

The low variance captured per component (even 3 PCs only reach 22.6%) reflects the high dimensionality of transcriptomics data — no single axis dominates. The PCA coloured by mutation status provides a visual sense of whether mutant/WT samples are linearly separable in expression space. Coloring by total RSEM (as a QC check) helps spot failed sequencing samples that would appear as outliers.

---

### 7. Key EDA Conclusions

| Finding | Implication for Preprocessing |
|---------|-------------------------------|
| Raw RSEM is right-skewed, range 0–3.3M | **Log₂(x+1) transform is required** |
| 4M structured NaNs (1,690 samples × 3,338 genes) | **Fill with 0** (batch-level absence = unexpressed) |
| 220+ genes with zero expression across all samples | **Remove zero-expression genes** |
| Wide CV distribution — many near-constant genes | **CV-based gene filter** (bottom 10% by CV) |
| Binary imbalance: 2:1 WT:mutant | Class weighting or SMOTE needed in modeling |
| Multi-class: Missense dominates (62%); rare classes <10 samples | Collapse rare classes into `Other`; consider dropping |

---

## Notebook 04 — TCGA Preprocessing

### Objective
Apply all transformations motivated by the EDA to produce a clean, model-ready feature matrix. No information from the test set is used — standardisation is explicitly deferred to model pipelines to prevent data leakage.

---

### Preprocessing Pipeline

#### Step 1 — Duplicate Patient Removal
- Checked for duplicate patient IDs after barcode truncation
- **Result: 0 duplicates** — no rows dropped

---

#### Step 2 — Log₂(x+1) Transformation
- Applied `log₂(RSEM + 1)` to all expression values
- **WARNING detected: 1,188,746 negative RSEM values** — these are artefacts of the EB++ batch correction algorithm (small negative values are expected and well-documented for this file). They are handled by clamping via the `fillna(0)` before log-transform.
- Post-transform range: **[-6.83, 22.76]**

---

#### Step 3 — Zero-Expression Gene Removal
- Genes with summed log-expression == 0 across all patients were dropped
- **Removed: 220 genes**
- **Genes remaining: 20,281**

---

#### Step 4 — CV-Based Gene Filter
- Coefficient of variation computed per gene on log₂-transformed values
- Bottom 10th percentile threshold: **CV = 0.0604**
- Genes below this threshold are near-constant across all patients — no predictive value
- **Removed: 2,028 genes**
- **Genes remaining: 18,253**

---

#### Step 5 — NaN Fill
- After log-transform (which used `fillna(0)` internally), **0 NaN values** remained
- The structured missingness identified in EDA was resolved by treating absence as zero expression

---

#### Step 6 — Multi-Class Label Harmonisation

Raw MAF Variant_Classification strings were collapsed into a cleaner label set:

| Raw MAF classification(s) | Final label | Count |
|--------------------------|-------------|-------|
| `Missense_Mutation` | `Missense` | 1,949 |
| `Nonsense_Mutation` | `Nonsense` | 452 |
| `Frame_Shift_Del` + `Frame_Shift_Ins` | `Frameshift` | 394 |
| `Splice_Site` | `Splice` | 215 |
| `In_Frame_Del` + `In_Frame_Ins` | `InFrame` | 68 |
| `Silent`, `3'UTR`, `5'UTR`, `Intron`, … | `Other` | 53 |
| `WT` | `WT` | 6,744 |

Grouping rationale:
- **Frameshift** merges deletions and insertions — both cause the same downstream effect (reading frame disruption and usually a truncated/non-functional protein)
- **InFrame** merges in-frame indels — both preserve the reading frame but alter the protein sequence locally
- **Other** catches functionally ambiguous or non-coding mutations; modeling notebooks will decide whether to keep this class or drop it

---

### Final Dataset

| Metric | Before | After |
|--------|--------|-------|
| Patients | 9,875 | 9,875 |
| Gene features | 20,501 | 18,253 |
| NaN values in features | 4,069,730 | 0 |
| Label columns | 2 | 3 (`Mutated`, `Variant_Classification`, `Mutation_Class`) |

#### Final binary label distribution

| Status | Count |
|--------|-------|
| Wild-type (WT) | 6,744 |
| Mutated | 3,131 |

#### Final multi-class label distribution

| Mutation_Class | Count | % of all patients |
|---------------|-------|------------------|
| WT | 6,744 | 68.3% |
| Missense | 1,949 | 19.7% |
| Nonsense | 452 | 4.6% |
| Frameshift | 394 | 4.0% |
| Splice | 215 | 2.2% |
| InFrame | 68 | 0.7% |
| Other | 53 | 0.5% |

**Output saved:** `data/processed/tcga_preprocessed.csv.gz` — 1,035.6 MB

---

## End-to-End Data Flow Summary

```
Raw TCGA files (raw/*)
        │
        ▼
02_tcga_data_loading.ipynb
  ├── Expression: 20,531 genes × 11,069 samples
  │     └─ Parse gene IDs → filter primary tumors → deduplicate → 9,875 × 20,501
  ├── Mutations: 3.6M events → filter TP53 PASS → resolve multi-hit → 3,325 patients labelled
  └── Merge (left join) → 9,875 × 20,503 → save tcga_merged_raw.csv.gz (697 MB)
        │
        ▼
03_tcga_eda.ipynb
  ├── Confirm: 4M structured NaNs, 1,690 affected samples
  ├── Confirm: RSEM right-skewed → log transform needed
  ├── Confirm: ~220 zero-expression genes → removal needed
  ├── Confirm: low-CV gene tail → CV filter needed
  ├── Confirm: binary 2:1 imbalance, multi-class Missense-heavy imbalance
  └── PCA: 3 PCs explain 22.6% variance on random gene subset
        │
        ▼
04_tcga_preprocessing.ipynb
  ├── Log₂(x+1) transform (note: 1.2M small negatives from EB++ → clamped to 0)
  ├── Remove 220 zero-expression genes → 20,281
  ├── Remove bottom-10%-CV genes (2,028) → 18,253
  ├── NaN fill (0 remaining after transform step)
  └── Harmonise labels → 7 final classes + save tcga_preprocessed.csv.gz (1,036 MB)
```

---

## Open Issues and Decisions for Modeling

| Issue | Status | Suggested Approach |
|-------|--------|--------------------|
| Binary class imbalance (2:1 WT:mutant) | Unresolved — defer to model pipeline | `class_weight='balanced'` or SMOTE on training folds only |
| Multi-class severe imbalance (Missense 62%) | Unresolved — defer to model pipeline | SMOTE; consider dropping `Other` (53 samples) or merging into WT |
| Rare classes: `InFrame` (68), `Other` (53) | Documented | May be too small for reliable classification; explore dropping or merging |
| Negative RSEM values from EB++ | Handled (clamped to 0) | Document as known artefact; consider flagging affected samples |
| Tissue/cohort confounding | Not yet addressed | Check whether TP53 mutation rate varies by tumor type; may need tissue-stratified analysis or tissue-type as a covariate |
| Feature standardisation | Explicitly deferred | Apply `StandardScaler` inside CV folds to prevent leakage |
| `Variant_Classification` column in final file | Retained alongside `Mutation_Class` | Drop before passing to any model |

---

## Numbers at a Glance

| Quantity | Value |
|----------|-------|
| Raw expression samples | 11,069 |
| Primary tumor samples retained | 9,875 |
| Raw gene features | 20,501 |
| Final gene features (after filtering) | 18,253 |
| Total TP53-mutant patients | 3,131 |
| Total wild-type patients | 6,744 |
| Missense (dominant mutation type) | 1,949 (62.2% of mutants) |
| Final file size | 1,035.6 MB |
