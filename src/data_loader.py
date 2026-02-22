"""
Data loading utilities for breast cancer prediction project.

Handles loading of:
- TCGA-BRCA preprocessed feature matrix
- GSE96058 / SCAN-B clinical data
- GSE96058 gene expression data (1.8GB)
"""

import pandas as pd
import numpy as np
import os


def load_tcga_feature_matrix(path="data/processed/02_tcga_feature_matrix.csv"):
    """
    Load the preprocessed TCGA-BRCA feature matrix.

    Returns DataFrame with columns:
        Pathway_Proliferation, Pathway_HER2_Signaling, Pathway_Estrogen_Response,
        Pathway_Immune_Response, Pathway_Invasion_EMT, Pathway_Apoptosis,
        Pathway_Angiogenesis, Ratio_Prolif_Apop, sample_id, high_risk,
        time_to_event, event_status
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"TCGA feature matrix not found at {path}. "
            "Ensure data/processed/02_tcga_feature_matrix.csv exists."
        )
    df = pd.read_csv(path)
    print(f"Loaded TCGA feature matrix: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def load_clinical_data(path):
    """
    Load clinical data CSV (works for both TCGA and GSE96058).

    Args:
        path: Path to clinical CSV file.

    Returns:
        DataFrame with clinical columns. Both files contain sample_id,
        time_to_event, event_status, and high_risk columns.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Clinical data not found at {path}")
    df = pd.read_csv(path)
    print(f"Loaded clinical data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df


def load_gse96058_expression(path="data/raw/GSE96058_gene_expression.csv"):
    """
    Load GSE96058 gene expression matrix.

    The raw file has genes as rows and samples (F1, F2, ...) as columns.
    This function transposes it to samples x genes format.

    Args:
        path: Path to the decompressed expression CSV.

    Returns:
        DataFrame with samples as rows and genes as columns,
        plus a 'sample_id' column.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"GSE96058 expression data not found at {path}. "
            "Run: cd data/raw && bash download_data.sh"
        )
    print("Loading GSE96058 expression data (this may take a moment)...")
    exp_df = pd.read_csv(path, index_col=0)

    # Transpose: genes as rows -> samples as rows
    exp_df = exp_df.T
    exp_df.index.name = "sample_id"
    exp_df = exp_df.reset_index()

    print(f"Loaded GSE96058 expression: {exp_df.shape[0]} samples, {exp_df.shape[1] - 1} genes")
    return exp_df
