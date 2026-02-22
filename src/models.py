"""
Model definitions and evaluation pipelines.

Implements three classifiers (Elastic Net, Random Forest, Gradient Boosting)
with stratified k-fold cross-validation and ablation study support.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score


def get_classifiers():
    """
    Return a dictionary of classifier constructors with paper-specified hyperparameters.

    Returns:
        Dict mapping model name to sklearn classifier instance.
    """
    return {
        "Elastic Net": LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            max_iter=5000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=500,
            learning_rate=0.03,
            max_depth=3,
            random_state=42,
        ),
    }


def run_cv_evaluation(X, y, classifiers=None, n_splits=5):
    """
    Evaluate classifiers using stratified k-fold cross-validation.

    StandardScaler is applied within each fold (fit on train, transform on test).

    Args:
        X: Feature matrix (DataFrame or ndarray).
        y: Binary labels.
        classifiers: Dict of name -> classifier. Defaults to get_classifiers().
        n_splits: Number of CV folds.

    Returns:
        DataFrame with columns: Model, AUC, AUC_SD, Accuracy
    """
    if classifiers is None:
        classifiers = get_classifiers()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_arr = np.array(X)
    y_arr = np.array(y)

    results = []

    for name, clf_template in classifiers.items():
        fold_aucs = []
        fold_accs = []

        for train_idx, test_idx in skf.split(X_arr, y_arr):
            X_train, X_test = X_arr[train_idx], X_arr[test_idx]
            y_train, y_test = y_arr[train_idx], y_arr[test_idx]

            # Scale within fold
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            # Clone classifier (fresh instance each fold)
            from sklearn.base import clone
            clf = clone(clf_template)
            clf.fit(X_train, y_train)

            y_proba = clf.predict_proba(X_test)[:, 1]
            y_pred = clf.predict(X_test)

            fold_aucs.append(roc_auc_score(y_test, y_proba))
            fold_accs.append(accuracy_score(y_test, y_pred))

        results.append({
            "Model": name,
            "AUC": np.mean(fold_aucs),
            "AUC_SD": np.std(fold_aucs),
            "Accuracy": np.mean(fold_accs),
        })

    return pd.DataFrame(results)


def run_ablation(pathway_features, clinical_features, y):
    """
    Run ablation study comparing pathway-only, clinical-only, and combined features.

    Each configuration is evaluated with 5-fold stratified CV.

    Args:
        pathway_features: DataFrame with 8 pathway features.
        clinical_features: DataFrame with 9 clinical features.
        y: Binary labels.

    Returns:
        DataFrame with columns: Feature Set, Model, AUC, AUC_SD, Accuracy
    """
    classifiers = get_classifiers()

    feature_sets = {
        f"Pathway Only ({pathway_features.shape[1]} features)": pathway_features,
        f"Clinical Only ({clinical_features.shape[1]} features)": clinical_features,
        f"Pathway + Clinical ({pathway_features.shape[1] + clinical_features.shape[1]} features)": pd.concat(
            [pathway_features.reset_index(drop=True),
             clinical_features.reset_index(drop=True)],
            axis=1,
        ),
    }

    all_results = []

    for fs_name, X in feature_sets.items():
        print(f"  Evaluating: {fs_name}")
        cv_results = run_cv_evaluation(X, y, classifiers)
        cv_results.insert(0, "Feature Set", fs_name)
        all_results.append(cv_results)

    return pd.concat(all_results, ignore_index=True)
