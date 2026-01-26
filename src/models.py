
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available. Will fallback to sklearn GradientBoosting.")
except Exception as e:
    XGB_AVAILABLE = False
    print(f"XGBoost failed to load: {e}. Will fallback to sklearn GradientBoosting.")

if not XGB_AVAILABLE:
    from sklearn.ensemble import GradientBoostingClassifier
from lifelines import CoxPHFitter
from typing import Any, Dict

def train_elastic_net(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Trains Elastic Net Logistic Regression."""
    print("Training Elastic Net...")
    # l1_ratio=0.5 means 50% Lasso, 50% Ridge
    model = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, max_iter=2000, class_weight='balanced')
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Trains Random Forest."""
    print("Training Random Forest...")
    # Reduce complexity for small data
    model = RandomForestClassifier(n_estimators=500, max_depth=7, min_samples_leaf=3, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> Any:
    """Trains XGBoost (or fallback)."""
    if XGB_AVAILABLE:
        print("Training XGBoost...")
        # Scale_pos_weight for imbalance
        ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', 
                                  scale_pos_weight=ratio, 
                                  max_depth=2, n_estimators=500, learning_rate=0.01, subsample=0.7,
                                  colsample_bytree=0.7,
                                  random_state=42)
        model.fit(X_train, y_train)
        return model
    else:
        print("Training Gradient Boosting (sklearn fallback)...")
        # Tuned for small data - very slow learning
        model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.005, 
                                           max_depth=2, subsample=0.6, 
                                           random_state=42)
        model.fit(X_train, y_train)
        return model

def train_cox_model(clinical_train: pd.DataFrame, expression_train: pd.DataFrame) -> Any:
    """
    Trains Cox Proportional Hazards model.
    Requires 'time_to_event' and 'event_status' in clinical data.
    """
    print("Training Cox PH Model...")
    # Combine data
    df = expression_train.copy()
    df['T'] = clinical_train['time_to_event'].values
    df['E'] = clinical_train['event_status'].values
    
    # CoxPH can fail with too many features relative to samples.
    # Usage of penalizer is recommended for high dim data
    cph = CoxPHFitter(penalizer=0.1)
    
    # For speed and stability in this high-dim synthesis, limit genes or use pathway scores
    # If using raw genes, it might assume too much memory/time. 
    # For this demo, we assume expression_train might already be reduced (e.g. pathway scores)
    # If features > 100, we might want to subset.
    
    try:
        cph.fit(df, duration_col='T', event_col='E')
        return cph
    except Exception as e:
        print(f"Cox training failed: {e}")
        return None
