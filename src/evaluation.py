
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, accuracy_score
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import shap
import os
from typing import Dict, Any

def calculate_metrics(y_true: np.array, y_pred: np.array, y_proba: np.array) -> Dict[str, float]:
    """
    Calculates performance metrics using optimal threshold (Youden's J).
    """
    auc = roc_auc_score(y_true, y_proba)
    
    # Find optimal threshold
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_thresh = thresholds[best_idx]
    
    # Recalculate preds
    y_pred_opt = (y_proba >= best_thresh).astype(int)
    
    acc = accuracy_score(y_true, y_pred_opt)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_opt).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return {
        "AUC": auc,
        "Accuracy": acc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Threshold": best_thresh
    }

def plot_roc_curve(y_true: np.array, y_proba: np.array, model_name: str, save_path: str):
    """Plots and saves ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_true, y_proba):.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.savefig(save_path)
    plt.close()

def plot_km_curves(clinical_df: pd.DataFrame, predictions: np.array, save_path: str):
    """
    Plots Kaplan-Meier survival curves for predicted high/low risk groups.
    predictions: binary (0=Low Risk, 1=High Risk)
    """
    kmf = KaplanMeierFitter()
    
    T = clinical_df['time_to_event']
    E = clinical_df['event_status']
    
    plt.figure(figsize=(10, 6))
    
    # High Risk
    idx_high = (predictions == 1)
    if idx_high.sum() > 0:
        kmf.fit(T[idx_high], event_observed=E[idx_high], label="Predicted High Risk")
        kmf.plot_survival_function()
        
    # Low Risk
    idx_low = (predictions == 0)
    if idx_low.sum() > 0:
        kmf.fit(T[idx_low], event_observed=E[idx_low], label="Predicted Low Risk")
        kmf.plot_survival_function()
        
    if idx_high.sum() > 0 and idx_low.sum() > 0:
        # Log-rank test
        results = logrank_test(T[idx_high], T[idx_low], event_observed_A=E[idx_high], event_observed_B=E[idx_low])
        plt.title(f'Survival Analysis by Predicted Risk (p-value={results.p_value:.4f})')
    else:
        plt.title('Survival Analysis (One group empty)')
        
    plt.xlabel('Time (Months)')
    plt.ylabel('Survival Probability')
    plt.savefig(save_path)
    plt.close()

def run_shap_analysis(model: Any, X_train: pd.DataFrame, X_test: pd.DataFrame, model_type: str, result_dir: str):
    """
    Runs SHAP analysis and saves plots.
    """
    print(f"Running SHAP analysis for {model_type}...")
    
    # Select explainer based on model type
    if model_type == 'XGBoost':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
    elif model_type == 'RandomForest':
        explainer = shap.TreeExplainer(model)
        # RF shap_values is a list for classes, take [1] for positive class
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
    else:
        # Linear / Generic
        # KernelExplainer is slow, LinearExplainer for linear models
        try:
            explainer = shap.LinearExplainer(model, X_train)
            shap_values = explainer.shap_values(X_test)
        except:
            print("Skipping SHAP for this model type (complex/unsupported).")
            return

    # Summary Plot
    plt.figure()
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(result_dir, f'shap_summary_{model_type}.png'), bbox_inches='tight')
    plt.close()
    
    # Bar plot (global importance)
    plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.savefig(os.path.join(result_dir, f'shap_importance_{model_type}.png'), bbox_inches='tight')
    plt.close()

def plot_gene_panel_curve(n_features: list, auc_scores: list, save_path: str):
    """
    Plots AUC vs Number of Genes.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(n_features, auc_scores, marker='o', linestyle='-')
    plt.xlabel('Number of Genes')
    plt.ylabel('AUC')
    plt.title('Minimal Gene Panel Discovery')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_subtype_performance(clin_df: pd.DataFrame, y_true: np.array, y_pred: np.array, save_path: str):
    """
    Plots accuracy/AUC per subtype.
    """
    if 'pam50_subtype' not in clin_df.columns:
        print("Subtype column not found for plotting.")
        return
        
    subtypes = clin_df['pam50_subtype'].unique()
    results = {}
    
    for sub in subtypes:
        if pd.isna(sub): continue
        idx = clin_df['pam50_subtype'] == sub
        if idx.sum() < 5: continue # Skip small groups
        
        y_sub_true = y_true[idx]
        y_sub_pred = y_pred[idx]
        acc = accuracy_score(y_sub_true, y_sub_pred)
        results[sub] = acc
        
    if not results:
        print("No valid subtype groups found.")
        return
        
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(results.keys()), y=list(results.values()))
    plt.ylabel('Accuracy')
    plt.title('Performance by PAM50 Subtype')
    plt.ylim(0, 1)
    plt.savefig(save_path)
    plt.close()
