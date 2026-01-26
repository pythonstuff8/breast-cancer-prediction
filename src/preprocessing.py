
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict

def clean_and_normalize(expression_df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans and normalizes gene expression data.
    Assumes sample_id is in the dataframe.
    """
    # Separate ID
    if 'sample_id' in expression_df.columns:
        ids = expression_df['sample_id']
        data = expression_df.drop('sample_id', axis=1)
    else:
        ids = expression_df.index
        data = expression_df
        
    # Handle missing values (simple imputation for now)
    data = data.fillna(data.mean())
    
    # Normalize (Z-score)
    scaler = StandardScaler()
    data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    
    # Reattach ID
    data_scaled['sample_id'] = ids.values
    
    return data_scaled

def prepare_datasets(expression_df: pd.DataFrame, clinical_df: pd.DataFrame, target_col: str = 'high_risk') -> Dict[str, pd.DataFrame]:
    """
    Merges data and splits into train/test sets.
    
    Returns:
        Dictionary with keys: 'X_train', 'X_test', 'y_train', 'y_test', 'train_clinical', 'test_clinical'
    """
    # Merge on sample_id
    merged = pd.merge(expression_df, clinical_df, on='sample_id')
    
    # Process Stage Feature
    if 'stage' in merged.columns:
        stage_map = {
            'Stage I': 1, 'Stage IA': 1, 'Stage IB': 1,
            'Stage II': 2, 'Stage IIA': 2, 'Stage IIB': 2,
            'Stage III': 3, 'Stage IIIA': 3, 'Stage IIIB': 3, 'Stage IIIC': 3,
            'Stage IV': 4,
            'Stage X': 0, '[Discrepancy]': 0
        }
        # Use 0 for unknown/NaN
        merged['stage_encoded'] = merged['stage'].map(stage_map).fillna(0)
    else:
        merged['stage_encoded'] = 0

    # Features (genes + stage) - exclude clinical columns and ID
    # We explicitly select expression columns (which should be pathway scores) + stage_encoded
    # Assuming expression_df passed here already contains features.
    
    # Identify feature columns: all numeric columns from expression_df + stage_encoded
    # But expression_df might have sample_id.
    feature_cols = [c for c in expression_df.columns if c != 'sample_id']
    
    # Update X to include stage_encoded
    X = merged[feature_cols].copy()
    # X['Stage_Clinical'] = merged['stage_encoded'] # Disabled: reduced performance
    
    y = merged[target_col]
    
    # Split
    X_train, X_test, y_train, y_test, clinical_train, clinical_test = train_test_split(
        X, y, merged, test_size=0.2, random_state=42, stratify=y
    )
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'clinical_train': clinical_train,
        'clinical_test': clinical_test
    }
