"""Load CCLE bulk RNA-seq expression and derive per-cell-line TP53 labels."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPR_FILE = "OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv"
MUTS_FILE = "OmicsSomaticMutations.csv"

# Known TP53 hotspot codons (Hess et al. 2019; COSMIC census)
TP53_HOTSPOT_CODONS = {
    "R175", "Y205", "R213", "G245", "R248", "R249",
    "R273", "R282", "Y220", "V157", "H179", "C176",
    "C135", "S241", "M237",
}

# MolecularConsequence SO term → fine-grained class
# Priority order: most severe wins when a cell line has multiple mutations
_MC_PRIORITY = [
    ("frameshift_variant",      "Frameshift"),
    ("splice_acceptor_variant", "Splice"),
    ("splice_donor_variant",    "Splice"),
    ("nonsense",                "Nonsense"),          # SO:0001587
    ("stop_gained",             "Nonsense"),
    ("start_lost",              "Nonsense"),
    ("initiator_codon_variant", "Nonsense"),
    ("inframe_insertion",       "InFrame"),
    ("inframe_deletion",        "InFrame"),
    ("inframe_indel",           "InFrame"),
    ("missense_variant",        "Missense"),
    # synonymous / silent → excluded (treated as WT-like)
    ("synonymous_variant",      None),
    ("stop_retained_variant",   None),
]


def _parse_terms(mc_values) -> set[str]:
    terms: set[str] = set()
    for mc in mc_values:
        for part in str(mc).split(","):
            terms.add(part.split("|")[-1].strip().strip('"').strip("'"))
    return terms


def _classify_tp53(mc_values) -> str | None:
    """Return fine-grained class for a set of MolecularConsequence values.

    Returns None when all consequences are synonymous/silent (to be excluded).
    """
    terms = _parse_terms(mc_values)
    for so_term, label in _MC_PRIORITY:
        if any(so_term in t for t in terms):
            return label
    return "Other"


def _is_hotspot(protein_changes) -> bool:
    """Return True if any protein change hits a known TP53 hotspot codon."""
    for pc in protein_changes:
        pc_str = str(pc)
        for codon in TP53_HOTSPOT_CODONS:
            if codon in pc_str:
                return True
    return False


def load_expression(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Cell-line × gene log2(TPM+1) matrix, indexed by ModelID."""
    raw = pd.read_csv(Path(data_dir) / EXPR_FILE, low_memory=False)
    # First few columns are metadata; ModelID is one of them
    meta_cols = ["SequencingID", "ModelConditionID", "ModelID",
                 "IsDefaultEntryForMC", "IsDefaultEntryForModel"]
    gene_cols = [c for c in raw.columns if c not in meta_cols and c != "Unnamed: 0"]
    expr = raw.set_index("ModelID")[gene_cols]
    # Keep only the default model condition per cell line (one row per ModelID)
    if "IsDefaultEntryForModel" in raw.columns:
        mask = raw["IsDefaultEntryForModel"] == "Yes"
        expr = raw[mask].set_index("ModelID")[gene_cols]
    return expr


def load_tp53_mutations(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """TP53-only somatic mutation calls, default model-condition rows."""
    muts_raw = pd.read_csv(Path(data_dir) / MUTS_FILE, low_memory=False)
    tp53 = muts_raw[muts_raw["HugoSymbol"] == "TP53"].copy()
    if "IsDefaultEntryForModel" in tp53.columns:
        tp53 = tp53[tp53["IsDefaultEntryForModel"] == "Yes"]
    return tp53.reset_index(drop=True)


def derive_labels(expr: pd.DataFrame, muts: pd.DataFrame) -> pd.DataFrame:
    """Per-cell-line TP53 labels.

    Columns:
      tp53_binary   — 0 (WT) / 1 (any non-silent mutation)
      tp53_class    — fine-grained: WT / Missense / Nonsense / Frameshift /
                      Splice / InFrame / Other
                      Silent mutations are excluded (cell line treated as WT).
      tp53_missense_binary — 0 (WT or non-Missense mutant) / 1 (Missense only)
      tp53_hotspot  — 0 / 1 (1 = hits a known TP53 hotspot codon)
    """
    tp53_class_per_cell = (
        muts.groupby("ModelID")["MolecularConsequence"].apply(_classify_tp53)
    )
    hotspot_per_cell = (
        muts.groupby("ModelID")["ProteinChange"].apply(_is_hotspot)
        if "ProteinChange" in muts.columns else pd.Series(dtype=bool)
    )

    labels = pd.DataFrame(index=expr.index)

    # Exclude cell lines whose only TP53 change is silent (tp53_class == None)
    non_silent_mutants = tp53_class_per_cell[tp53_class_per_cell.notna()].index
    labels["tp53_binary"] = labels.index.isin(non_silent_mutants).astype(int)
    labels["tp53_class"] = labels.index.map(tp53_class_per_cell).fillna("WT")
    # Replace None (silent-only) with "WT" so downstream code stays clean
    labels["tp53_class"] = labels["tp53_class"].replace({None: "WT"})

    labels["tp53_missense_binary"] = (labels["tp53_class"] == "Missense").astype(int)
    labels["tp53_hotspot"] = labels.index.map(hotspot_per_cell).fillna(False).astype(int)

    return labels


def load_ccle(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (expression, labels) aligned by ModelID."""
    expr = load_expression(data_dir)
    muts = load_tp53_mutations(data_dir)
    labels = derive_labels(expr, muts)
    return expr, labels
