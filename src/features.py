"""
Feature engineering for pathway-level gene expression scores.

Defines 7 biological pathways (45 genes total) and computes
pathway activity scores as the mean of z-score normalized
constituent gene expression values.
"""

import pandas as pd
import numpy as np

# Seven biological pathways and their constituent genes
PATHWAY_DEFINITIONS = {
    "Proliferation": [
        "MKI67", "AURKA", "BIRC5", "CCNB1", "MYBL2",
        "PLK1", "BUB1", "CDC20", "MCM2", "PCNA",
    ],
    "Estrogen_Response": [
        "ESR1", "PGR", "FOXA1", "GATA3", "XBP1", "CA12",
    ],
    "Immune_Response": [
        "CD8A", "CD4", "PDCD1", "CD274", "CTLA4", "FOXP3", "IL2RA",
    ],
    "Invasion_EMT": [
        "VIM", "CDH2", "FN1", "MMP9", "TWIST1", "SNAI1", "ZEB1",
    ],
    "Apoptosis": [
        "CASP3", "CASP8", "BAX", "BAK1", "BAD", "TP53",
    ],
    "HER2_Signaling": [
        "ERBB2", "GRB7", "PGAP3", "MIEN1", "STARD3",
    ],
    "Angiogenesis": [
        "VEGFA", "KDR", "FLT1", "FGF2",
    ],
}


def compute_pathway_scores(expression_df, pathway_dict=None):
    """
    Compute pathway activity scores from z-score normalized expression.

    For each pathway, the score is the mean expression across its
    constituent genes. The input expression data should already be
    z-score normalized (per gene, across samples).

    Args:
        expression_df: Z-score normalized expression DataFrame
            (samples x genes). May include 'sample_id' column.
        pathway_dict: Dict mapping pathway names to gene lists.
            Defaults to PATHWAY_DEFINITIONS.

    Returns:
        DataFrame with one column per pathway (Pathway_<name>),
        indexed to match input.
    """
    if pathway_dict is None:
        pathway_dict = PATHWAY_DEFINITIONS

    scores = pd.DataFrame(index=expression_df.index)

    for name, genes in pathway_dict.items():
        valid_genes = [g for g in genes if g in expression_df.columns]
        missing_genes = [g for g in genes if g not in expression_df.columns]

        if missing_genes:
            print(f"  {name}: {len(missing_genes)} genes not found: {missing_genes}")

        if valid_genes:
            scores[f"Pathway_{name}"] = expression_df[valid_genes].mean(axis=1)
        else:
            print(f"  WARNING: No genes found for pathway {name}")
            scores[f"Pathway_{name}"] = 0.0

    return scores


def add_ratio_features(pathway_df):
    """
    Add engineered ratio features to pathway scores.

    Adds: Ratio_Prolif_Apop = Pathway_Proliferation - Pathway_Apoptosis

    Args:
        pathway_df: DataFrame with pathway score columns.

    Returns:
        DataFrame with ratio feature added.
    """
    result = pathway_df.copy()
    if "Pathway_Proliferation" in result.columns and "Pathway_Apoptosis" in result.columns:
        result["Ratio_Prolif_Apop"] = (
            result["Pathway_Proliferation"] - result["Pathway_Apoptosis"]
        )
    return result


def build_feature_matrix(pathway_scores, clinical_features=None):
    """
    Combine pathway scores and clinical features into a single matrix.

    Args:
        pathway_scores: DataFrame with pathway score columns
            (8 features: 7 pathways + ratio).
        clinical_features: Optional DataFrame with encoded clinical
            feature columns (9 features). Must have matching index.

    Returns:
        Combined DataFrame (8 or 17 features).
    """
    if clinical_features is not None:
        combined = pd.concat(
            [pathway_scores.reset_index(drop=True),
             clinical_features.reset_index(drop=True)],
            axis=1,
        )
        return combined
    return pathway_scores.copy()
