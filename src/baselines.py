"""
Baseline scoring methods for pathway activity quantification.

Implements a simplified rank-based ssGSEA (single-sample Gene Set
Enrichment Analysis) for comparison against mean z-score pathway scores.
"""

import pandas as pd
import numpy as np
from src.features import PATHWAY_DEFINITIONS
from src.models import run_cv_evaluation, get_classifiers
from src.survival import cox_cv


def compute_ssgsea_scores(expression_df, pathway_dict=None):
    """
    Compute ssGSEA-style pathway scores using rank-based enrichment.

    For each sample, genes are ranked by expression. The enrichment
    score for each pathway is computed as:
        ES = (sum_of_ranks_in_set - expected_rank_sum) / normalization

    This is a simplified rank-based approximation (not the full
    Barbie et al. implementation) that serves as a reasonable baseline.

    Args:
        expression_df: Expression DataFrame (samples x genes).
            May include 'sample_id' column.
        pathway_dict: Dict mapping pathway names to gene lists.
            Defaults to PATHWAY_DEFINITIONS.

    Returns:
        DataFrame with one column per pathway (Pathway_<name>),
        indexed to match input.
    """
    if pathway_dict is None:
        pathway_dict = PATHWAY_DEFINITIONS

    if "sample_id" in expression_df.columns:
        data = expression_df.drop("sample_id", axis=1)
    else:
        data = expression_df

    n_total = data.shape[1]  # total number of genes
    scores = pd.DataFrame(index=expression_df.index)

    for name, genes in pathway_dict.items():
        valid_genes = [g for g in genes if g in data.columns]
        if not valid_genes:
            scores[f"Pathway_{name}"] = 0.0
            continue

        n_set = len(valid_genes)
        # Expected rank sum under null: n_set * (n_total + 1) / 2
        expected_rank_sum = n_set * (n_total + 1) / 2
        normalizer = np.sqrt(n_total * n_set)

        pathway_scores = []
        for idx in data.index:
            sample = data.loc[idx]
            ranks = sample.rank(ascending=True)
            rank_sum = ranks[valid_genes].sum()
            es = (rank_sum - expected_rank_sum) / normalizer
            pathway_scores.append(es)

        scores[f"Pathway_{name}"] = pathway_scores

    return scores


def compare_scoring_methods(meanz_scores, ssgsea_scores, y,
                            time=None, event=None, classifiers=None):
    """
    Compare mean-z and ssGSEA pathway scoring methods.

    Runs the same classifiers (and optionally Cox model) on both
    scoring methods under identical 5-fold CV.

    Args:
        meanz_scores: DataFrame of mean-z pathway scores (8 features).
        ssgsea_scores: DataFrame of ssGSEA pathway scores (8 features).
        y: Binary labels.
        time: Optional time-to-event for Cox comparison.
        event: Optional event indicator for Cox comparison.
        classifiers: Dict of classifiers. Defaults to get_classifiers().

    Returns:
        DataFrame with columns: Model, Mean_Z_AUC, ssGSEA_AUC, Delta
    """
    if classifiers is None:
        classifiers = get_classifiers()

    meanz_results = run_cv_evaluation(meanz_scores, y, classifiers)
    ssgsea_results = run_cv_evaluation(ssgsea_scores, y, classifiers)

    comparison = pd.DataFrame({
        "Model": meanz_results["Model"],
        "Mean_Z_AUC": meanz_results["AUC"],
        "ssGSEA_AUC": ssgsea_results["AUC"],
        "Delta": meanz_results["AUC"] - ssgsea_results["AUC"],
    })

    # Cox comparison if survival data provided
    if time is not None and event is not None:
        meanz_cindex, _, _ = cox_cv(meanz_scores, time, event)
        ssgsea_cindex, _, _ = cox_cv(ssgsea_scores, time, event)
        cox_row = pd.DataFrame([{
            "Model": "Cox PH",
            "Mean_Z_AUC": meanz_cindex,
            "ssGSEA_AUC": ssgsea_cindex,
            "Delta": meanz_cindex - ssgsea_cindex,
        }])
        comparison = pd.concat([comparison, cox_row], ignore_index=True)

    return comparison
