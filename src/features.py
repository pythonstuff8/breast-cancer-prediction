
import pandas as pd
import numpy as np
from typing import Dict, List

def define_real_pathways() -> Dict[str, List[str]]:
    """
    Defines real gene sets (approximate Hallmark/PAM50).
    """
    pathways = {
        "Proliferation": ['MKI67', 'AURKA', 'BIRC5', 'CCNB1', 'MYBL2', 'PLK1', 'BUB1', 'CDC20', 'MCM2', 'PCNA'],
        "HER2_Signaling": ['ERBB2', 'GRB7', 'PGAP3', 'MIEN1', 'STARD3'],
        "Estrogen_Response": ['ESR1', 'PGR', 'FOXA1', 'GATA3', 'XBP1', 'CA12'],
        "Immune_Response": ['CD8A', 'CD4', 'PDCD1', 'CD274', 'CTLA4', 'FOXP3', 'IL2RA'],
        "Invasion_EMT": ['VIM', 'CDH2', 'FN1', 'MMP9', 'TWIST1', 'SNAI1', 'ZEB1'],
        "Apoptosis": ['CASP3', 'CASP8', 'BAX', 'BAK1', 'BAD', 'TP53'],
        "Angiogenesis": ['VEGFA', 'KDR', 'FLT1', 'FGF2']
    }
    return pathways

def calculate_pathway_scores(expression_df: pd.DataFrame, pathways: Dict[str, List[str]] = None) -> pd.DataFrame:
    """
    Calculates simple Single Sample Mean scores for pathways.
    """
    if pathways is None:
        # Check if synthetic
        cols = expression_df.columns.tolist()
        if "Gene_0" in cols:
             # Fallback to old synthetic logic if needed, or just warn
             # For now, let's assume we want real if not provided
             print("Feature extraction: Using real biological pathways.")
             pathways = define_real_pathways()
        else:
             print("Feature extraction: Using real biological pathways.")
             pathways = define_real_pathways()
            
    scores = pd.DataFrame(index=expression_df.index)
    
    for name, genes in pathways.items():
        # Only use genes present in the data
        valid_genes = [g for g in genes if g in expression_df.columns]
        if valid_genes:
            # Z-score within sample?
            # Standard approach: just mean of normalized expression
            scores[f"Pathway_{name}"] = expression_df[valid_genes].mean(axis=1)
            
    return scores

def add_top_variant_genes(expression_df: pd.DataFrame, n_top: int = 50) -> pd.DataFrame:
    """
    Selects top N genes with highest variance across samples (most informative).
    """
    # Exclude non-gene columns
    gene_cols = [c for c in expression_df.columns if c != 'sample_id' and not c.startswith('Pathway_')]
    
    if not gene_cols:
        return pd.DataFrame(index=expression_df.index)
        
    variances = expression_df[gene_cols].var()
    top_genes = variances.nlargest(n_top).index.tolist()
    
    return expression_df[top_genes]

def add_ratio_features(feature_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds ratio based features (e.g., Proliferation / Apoptosis).
    """
    if "Pathway_Proliferation" in feature_df.columns and "Pathway_Apoptosis" in feature_df.columns:
        feature_df["Ratio_Prolif_Apop"] = feature_df["Pathway_Proliferation"] - feature_df["Pathway_Apoptosis"]
        
    return feature_df
