"""
Cox proportional hazards survival analysis.

Implements Cox PH model fitting, cross-validated C-index evaluation,
and Kaplan-Meier visualization with log-rank testing.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def fit_cox_model(X, time, event, penalizer=0.1):
    """
    Fit a Cox proportional hazards model.

    Args:
        X: Feature matrix (DataFrame).
        time: Time-to-event values (Series or array).
        event: Event indicator (1=event, 0=censored).
        penalizer: L2 regularization strength.

    Returns:
        Fitted CoxPHFitter instance.
    """
    df = X.copy()
    df["T"] = np.array(time)
    df["E"] = np.array(event)

    cph = CoxPHFitter(penalizer=penalizer)
    cph.fit(df, duration_col="T", event_col="E")
    return cph


def cox_cv(X, time, event, n_splits=5, penalizer=0.1):
    """
    Evaluate Cox model using stratified k-fold cross-validation.

    C-index is computed on each test fold. Stratification is on the
    binary event indicator.

    Args:
        X: Feature matrix (DataFrame or ndarray).
        time: Time-to-event values.
        event: Event indicator (binary).
        n_splits: Number of CV folds.
        penalizer: Cox PH regularization.

    Returns:
        Tuple of (mean_cindex, std_cindex, list_of_fold_cindexes).
    """
    X_arr = np.array(X)
    time_arr = np.array(time)
    event_arr = np.array(event).astype(int)
    feature_names = X.columns.tolist() if hasattr(X, "columns") else [f"f{i}" for i in range(X_arr.shape[1])]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_cindexes = []

    for train_idx, test_idx in skf.split(X_arr, event_arr):
        X_train, X_test = X_arr[train_idx], X_arr[test_idx]
        t_train, t_test = time_arr[train_idx], time_arr[test_idx]
        e_train, e_test = event_arr[train_idx], event_arr[test_idx]

        # Standardize within fold
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Fit Cox on training fold
        train_df = pd.DataFrame(X_train_s, columns=feature_names)
        train_df["T"] = t_train
        train_df["E"] = e_train

        cph = CoxPHFitter(penalizer=penalizer)
        cph.fit(train_df, duration_col="T", event_col="E")

        # Evaluate on test fold
        test_df = pd.DataFrame(X_test_s, columns=feature_names)
        test_df["T"] = t_test
        test_df["E"] = e_test

        c_index = cph.score(test_df, scoring_method="concordance_index")
        fold_cindexes.append(c_index)

    return np.mean(fold_cindexes), np.std(fold_cindexes), fold_cindexes


def generate_km_curves(X, time, event, model, save_path=None):
    """
    Generate Kaplan-Meier survival curves for Cox-predicted risk groups.

    Patients are split at the median predicted partial hazard into
    high-risk and low-risk groups. Includes log-rank test p-value.

    Args:
        X: Feature matrix (DataFrame).
        time: Time-to-event values.
        event: Event indicator.
        model: Fitted CoxPHFitter.
        save_path: Optional path to save the figure.

    Returns:
        matplotlib Figure, log-rank test p-value.
    """
    # Predict partial hazards
    partial_hazards = model.predict_partial_hazard(X)
    median_hazard = partial_hazards.median()

    high_risk = partial_hazards >= median_hazard
    low_risk = ~high_risk

    time_arr = np.array(time)
    event_arr = np.array(event)

    # Filter to positive times
    valid = time_arr > 0
    time_arr = time_arr[valid]
    event_arr = event_arr[valid]
    high_risk = high_risk[valid]
    low_risk = low_risk[valid]

    fig, ax = plt.subplots(figsize=(8, 6))
    kmf = KaplanMeierFitter()

    # High risk group
    kmf.fit(
        time_arr[high_risk],
        event_observed=event_arr[high_risk],
        label="High Risk",
    )
    kmf.plot_survival_function(ax=ax, ci_show=True, color="red")

    # Low risk group
    kmf.fit(
        time_arr[low_risk],
        event_observed=event_arr[low_risk],
        label="Low Risk",
    )
    kmf.plot_survival_function(ax=ax, ci_show=True, color="blue")

    # Log-rank test
    lr_result = logrank_test(
        time_arr[high_risk], time_arr[low_risk],
        event_observed_A=event_arr[high_risk],
        event_observed_B=event_arr[low_risk],
    )

    ax.set_xlabel("Time (months)", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_title(
        f"Kaplan-Meier Survival Curves by Predicted Risk Group\n"
        f"Log-rank p = {lr_result.p_value:.2e}",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved KM curves to {save_path}")

    return fig, lr_result.p_value
