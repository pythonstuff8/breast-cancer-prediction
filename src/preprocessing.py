"""
Preprocessing utilities for gene expression and clinical data.

Includes z-score normalization, clinical feature encoding,
and outcome variable filtering.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def zscore_normalize(expression_df):
    """
    Z-score normalize gene expression data across samples (per gene).

    For each gene: z = (value - mean) / std, computed across all samples
    within the cohort.

    Args:
        expression_df: DataFrame with samples as rows, genes as columns.
            May include a 'sample_id' column which is preserved.

    Returns:
        DataFrame with z-score normalized expression values.
    """
    if "sample_id" in expression_df.columns:
        ids = expression_df["sample_id"].copy()
        data = expression_df.drop("sample_id", axis=1)
    else:
        ids = None
        data = expression_df.copy()

    # Z-score per gene (column-wise): (x - mean) / std
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    stds = stds.replace(0, 1)  # avoid division by zero for constant genes
    normalized = (data - means) / stds

    if ids is not None:
        normalized.insert(0, "sample_id", ids.values)

    return normalized


def encode_clinical_features(clinical_df):
    """
    Encode clinical features for model input.

    Categorical features are encoded with LabelEncoder (NaN -> "Unknown").
    Numeric features are imputed with median.

    Args:
        clinical_df: GSE96058 clinical DataFrame.

    Returns:
        DataFrame with encoded feature columns:
            lymph_node_status_enc, er_status_enc, pgr_status_enc,
            her2_status_enc, ki67_status_enc, nhg_enc, pam50_subtype_enc,
            age_at_diagnosis, tumor_size
    """
    result = pd.DataFrame(index=clinical_df.index)

    # Categorical features to encode
    categorical_cols = {
        "lymph_node_status": "lymph_node_status_enc",
        "er_status": "er_status_enc",
        "pgr_status": "pgr_status_enc",
        "her2_status": "her2_status_enc",
        "ki67_status": "ki67_status_enc",
        "nhg": "nhg_enc",
        "pam50_subtype": "pam50_subtype_enc",
    }

    for src_col, dst_col in categorical_cols.items():
        if src_col in clinical_df.columns:
            col = clinical_df[src_col].fillna("Unknown").astype(str)
            le = LabelEncoder()
            result[dst_col] = le.fit_transform(col)
        else:
            print(f"Warning: column '{src_col}' not found in clinical data")
            result[dst_col] = 0

    # Numeric features (impute with median)
    for col in ["age_at_diagnosis", "tumor_size"]:
        if col in clinical_df.columns:
            result[col] = pd.to_numeric(clinical_df[col], errors="coerce")
            result[col] = result[col].fillna(result[col].median())
        else:
            print(f"Warning: column '{col}' not found in clinical data")
            result[col] = 0

    return result


def filter_outcome(clinical_df):
    """
    Filter to patients with a defined binary outcome (high_risk).

    Keeps only rows where high_risk is not null.

    Args:
        clinical_df: DataFrame with 'high_risk' column.

    Returns:
        Filtered DataFrame.
    """
    if "high_risk" not in clinical_df.columns:
        raise ValueError("'high_risk' column not found in clinical data")

    filtered = clinical_df.dropna(subset=["high_risk"]).copy()
    filtered["high_risk"] = filtered["high_risk"].astype(int)
    print(f"Filtered to {len(filtered)} patients with defined outcome "
          f"(high_risk=1: {filtered['high_risk'].sum()}, "
          f"high_risk=0: {(filtered['high_risk'] == 0).sum()})")
    return filtered
