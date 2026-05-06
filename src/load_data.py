"""Load CCLE bulk RNA-seq expression and derive per-cell-line TP53 labels."""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw"

EXPR_FILE = "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
MUTS_FILE = "OmicsSomaticMutations.csv"


def _classify_tp53(mc_values) -> str:
    # Priority: Frame_Shift_Del > Splice_Site > Missense_Mutation > Other
    terms: set[str] = set()
    for mc in mc_values:
        for part in str(mc).split(","):
            terms.add(part.split("|")[-1].strip())
    if any("frameshift" in t for t in terms):
        return "Frame_Shift_Del"
    if any(t in ("splice_donor_variant", "splice_acceptor_variant") for t in terms):
        return "Splice_Site"
    if any("missense" in t for t in terms):
        return "Missense_Mutation"
    return "Other"


def load_expression(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """Cell-line × gene log2(TPM+1) matrix, indexed by ModelID. 24Q4 file is pre-filtered."""
    return pd.read_csv(Path(data_dir) / EXPR_FILE, index_col=0)


def load_tp53_mutations(data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    """TP53-only somatic mutation calls."""
    muts_raw = pd.read_csv(Path(data_dir) / MUTS_FILE, low_memory=False)
    return muts_raw[muts_raw["HugoSymbol"] == "TP53"].reset_index(drop=True)


def derive_labels(expr: pd.DataFrame, muts: pd.DataFrame) -> pd.DataFrame:
    """Per-cell-line TP53 labels: tp53_binary (0/1) and tp53_class (5 categories)."""
    tp53_class_per_cell = (
        muts.groupby("ModelID")["MolecularConsequence"].apply(_classify_tp53)
    )
    labels = pd.DataFrame(index=expr.index)
    labels["tp53_binary"] = labels.index.isin(tp53_class_per_cell.index).astype(int)
    labels["tp53_class"] = labels.index.map(tp53_class_per_cell).fillna("WT")
    return labels


def load_ccle(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper returning (expression, labels) aligned by ModelID."""
    expr = load_expression(data_dir)
    muts = load_tp53_mutations(data_dir)
    labels = derive_labels(expr, muts)
    return expr, labels
