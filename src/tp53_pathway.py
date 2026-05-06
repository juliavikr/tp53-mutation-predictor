"""Curated TP53 pathway gene set (HGNC symbols), used to annotate top model features.

Sources: KEGG hsa04115 (p53 signaling), MSigDB Hallmark P53 pathway, canonical reviews
(Vousden & Prives 2009; Kastenhuber & Lowe 2017). Conservative list focused on direct
TP53 transcriptional targets, immediate regulators, and core apoptosis/cell-cycle effectors.
"""

TP53_DIRECT_TARGETS = {
    # Cell-cycle arrest
    "CDKN1A", "GADD45A", "GADD45B", "GADD45G", "SFN", "CCNG1", "BTG2",
    # Apoptosis (intrinsic)
    "BAX", "BBC3", "PMAIP1", "BCL2L11", "BID", "TP53AIP1", "PERP", "SIVA1",
    "PHLDA3",
    # Apoptosis (extrinsic, death receptors)
    "FAS", "TNFRSF10A", "TNFRSF10B", "TNFRSF10D",
    # DNA repair / stress
    "DDB2", "XPC", "RRM2B", "RAD51", "TP53I3", "TP53INP1", "ZMAT3",
    # Senescence / metabolism
    "IGFBP3", "SERPINB5", "THBS1", "SESN1", "SESN2", "TIGAR", "GLS2", "SCO2",
    # Cytoskeleton / actin (Rac antagonist)
    "CYFIP2",
}

TP53_REGULATORS = {
    "MDM2", "MDM4", "ATM", "ATR", "CHEK1", "CHEK2",
    "PPM1D", "HIPK2", "USP7", "DAXX",
}

TP53_FAMILY = {"TP53", "TP63", "TP73"}

# Wider apoptosis / cell cycle context (often correlated with TP53 status)
WIDER_PATHWAY = {
    "BCL2", "BCL2L1", "MCL1", "BAK1",
    "CDKN2A", "CDKN2B", "RB1",
    "CCND1", "CCNE1", "CDK2", "CDK4", "CDK6",
    "MYC",
}

# Combined set — what we cross-reference against
TP53_PATHWAY: set[str] = (
    TP53_DIRECT_TARGETS | TP53_REGULATORS | TP53_FAMILY | WIDER_PATHWAY
)


def annotate(symbols: list[str]) -> list[dict]:
    """Tag each symbol with which sub-set it belongs to (or 'other')."""
    out = []
    for s in symbols:
        if s in TP53_FAMILY:
            cat = "TP53 family"
        elif s in TP53_REGULATORS:
            cat = "TP53 regulator"
        elif s in TP53_DIRECT_TARGETS:
            cat = "TP53 direct target"
        elif s in WIDER_PATHWAY:
            cat = "wider pathway"
        else:
            cat = "other"
        out.append({"gene_symbol": s, "tp53_category": cat,
                    "in_tp53_pathway": cat != "other"})
    return out
