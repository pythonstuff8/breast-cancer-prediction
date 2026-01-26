
import pandas as pd
import numpy as np
import os
from typing import Tuple, Optional

def generate_synthetic_data(n_samples: int = 500, n_genes: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic gene expression data and clinical data for testing.
    
    Args:
        n_samples: Number of patients to simulate.
        n_genes: Number of genes to simulate.
        
    Returns:
        Tuple of (expression_df, clinical_df)
    """
    print(f"Generating synthetic data: {n_samples} samples, {n_genes} genes...")
    
    # Simulate gene expression (log-normal distribution then log-transformed ~ normal)
    # We'll make some genes correlated with the outcome to ensure models can learn something
    expression_data = np.random.normal(loc=10, scale=2, size=(n_samples, n_genes))
    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    expression_df = pd.DataFrame(expression_data, columns=gene_names)
    
    # Simulate clinical data
    clinical_data = pd.DataFrame({
        'sample_id': [f"Patient_{i}" for i in range(n_samples)],
        'age': np.random.randint(30, 80, n_samples),
        'stage': np.random.choice(['Stage I', 'Stage II', 'Stage III', 'Stage IV'], n_samples),
        'pam50_subtype': np.random.choice(['Luminal A', 'Luminal B', 'Basal', 'HER2'], n_samples),
        'time_to_event': np.random.exponential(scale=60, size=n_samples), # Months
        'event_status': np.random.choice([0, 1], n_samples, p=[0.7, 0.3]) # 1 = recurrence/death
    })
    
    # Inject signal: Make outcome dependent on some "important" genes
    # Let's say Gene_0 to Gene_9 are "proliferation" genes effectively
    risk_score = np.mean(expression_data[:, :10], axis=1)
    
    # strong signal: normalize risk score to be centered around 1, but with wider variance
    risk_factor = (risk_score - 10) / 2 # ~ N(0, 1)
    # Sigmoid to probability
    prob_death = 1 / (1 + np.exp(-(risk_factor + 0.5))) # shift slightly to have enough events
    
    # Event status based on risk
    clinical_data['event_status'] = np.random.binomial(1, prob_death)
    
    # Time to event: lower for high risk
    # Base time ~ Exp(60)
    base_time = np.random.exponential(scale=60, size=n_samples)
    # If high risk (risk_factor > 0), reduce time significantly
    modifier = np.exp(-risk_factor) # e^-x: if x=1 (high risk), mod=0.36. if x=-1 (low), mod=2.7
    clinical_data['time_to_event'] = base_time * modifier
    
    # Define binary risk from time_to_event for classification tasks (e.g., event within 5 years)
    # 60 months = 5 years
    clinical_data['high_risk'] = ((clinical_data['time_to_event'] < 60) & (clinical_data['event_status'] == 1)).astype(int)
    
    expression_df['sample_id'] = clinical_data['sample_id']
    
    return expression_df, clinical_data

def load_real_tcga_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads real TCGA-BRCA data from UCSC Xena files.
    """
    print("Loading real TCGA data...")
    exp_path = os.path.join(data_dir, "TCGA.BRCA.sampleMap_HiSeqV2")
    clin_path = os.path.join(data_dir, "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix")
    
    if not os.path.exists(exp_path) or not os.path.exists(clin_path):
        raise FileNotFoundError(f"Data files not found in {data_dir}. Please run download command.")

    # Load Clinical Data
    # 'OS_Time_nature2012': Days to death or last follow up
    # 'OS_event_nature2012': 1=Death, 0=Alive
    clin_df = pd.read_csv(clin_path, sep='\t')
    
    # Filter for samples with survival data
    clin_df = clin_df.dropna(subset=['OS_Time_nature2012', 'OS_event_nature2012'])
    
    # Standardize columns
    clin_df['sample_id'] = clin_df['sampleID']
    clin_df['time_to_event'] = clin_df['OS_Time_nature2012'] / 30.0 # Convert to months roughly
    clin_df['event_status'] = clin_df['OS_event_nature2012'].astype(int)
    
    # Extract Stage (if available)
    if 'AJCC_Stage_nature2012' in clin_df.columns:
        clin_df['stage'] = clin_df['AJCC_Stage_nature2012']
    elif 'pathologic_stage' in clin_df.columns:
        clin_df['stage'] = clin_df['pathologic_stage']
    else:
        clin_df['stage'] = np.nan
    
    # Extract PAM50 Subtype
    if 'PAM50Call_RNAseq' in clin_df.columns:
        clin_df['pam50_subtype'] = clin_df['PAM50Call_RNAseq']
    elif 'PAM50_mRNA_nature2012' in clin_df.columns:
         clin_df['pam50_subtype'] = clin_df['PAM50_mRNA_nature2012']
    else:
         clin_df['pam50_subtype'] = np.nan

    # Create binary high_risk outcome (e.g., Death within 5 years = 60 months)
    # Note: Censored patients (event=0) with time < 60 cannot be confirmed as low risk (unknown).
    # We will exclude them or count them as low risk for simplification, but correct is exclusion.
    # For robust training, let's keep it simple: 
    # High Risk = Death within 5 years.
    # Low Risk = Alive > 5 years.
    # Exclude: Alive < 5 years (censored early).
    
    clin_df['high_risk'] = np.nan
    clin_df.loc[(clin_df['event_status'] == 1) & (clin_df['time_to_event'] < 60), 'high_risk'] = 1
    clin_df.loc[(clin_df['time_to_event'] >= 60), 'high_risk'] = 0
    
    # Drop samples with indeterminate binary outcome
    clin_df = clin_df.dropna(subset=['high_risk'])
    clin_df['high_risk'] = clin_df['high_risk'].astype(int)
    
    print(f"Clinical samples after filtering: {len(clin_df)}")
    
    # Load Expression Data
    # Rows=Genes, Cols=Samples
    # We transpose to Samples x Genes
    exp_df = pd.read_csv(exp_path, sep='\t', index_col=0)
    exp_df = exp_df.T
    exp_df.index.name = 'sample_id'
    exp_df = exp_df.reset_index()
    
    # Intersect samples
    common_samples = list(set(clin_df['sample_id']) & set(exp_df['sample_id']))
    print(f"Common samples found: {len(common_samples)}")
    
    exp_df = exp_df[exp_df['sample_id'].isin(common_samples)]
    clin_df = clin_df[clin_df['sample_id'].isin(common_samples)]
    
    return exp_df, clin_df

def load_gse96058_data(data_dir: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads GSE96058 data (SCAN-B) for external validation.
    """
    print("Loading GSE96058 (External Validation) data...")
    exp_path = os.path.join(data_dir, "GSE96058_expression.csv.gz")
    clin_path = os.path.join(data_dir, "GSE96058_series_matrix.txt")
    
    if not os.path.exists(clin_path):
        # Allow loading gz
        clin_path_gz = clin_path + ".gz"
        if os.path.exists(clin_path_gz):
            import gzip
            # We need to read it to find metadata
            clin_path = clin_path_gz # simpler to handle
        else:
            raise FileNotFoundError("GSE96058 clinical file not found.")

    # Parse Series Matrix for Clinical Data
    # This file is row-oriented (Attributes x Samples)
    # We need to parse !Sample_geo_accession and !Sample_characteristics_ch1
    
    meta_dict = {}
    samples = []
    
    # Simple manual parser
    import gzip
    open_func = gzip.open if clin_path.endswith('.gz') else open
    mode = 'rt' if clin_path.endswith('.gz') else 'r'
    
    characteristics = []
    
    with open_func(clin_path, mode, errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_title'):
                 # These matches columns in Expression CSV ("F1", "F2", ...)
                parts = line.strip().replace('"', '').split('\t')
                samples = parts[1:] 
            elif line.startswith('!Sample_characteristics_ch1'):
                parts = line.strip().replace('"', '').split('\t')
                characteristics.append(parts[1:])
            elif line.startswith('!series_matrix_table_begin'):
                break

    # Convert to DataFrame
    clin_data = {}
    clin_data['sample_id'] = samples
    
    for row in characteristics:
        # Check first element to guess feature type
        # Ideally we parse "key: value"
        key_name = row[0].split(':')[0].strip() if ':' in row[0] else "Unknown"
        values = []
        for item in row:
            if ':' in item:
                val = item.split(':', 1)[1].strip()
            else:
                val = item
            values.append(val)
        
        # Duplicate keys? Append index
        if key_name in clin_data:
            key_name = f"{key_name}_2"
        clin_data[key_name] = values
        
    clin_df = pd.DataFrame(clin_data)
    
    # Process survival
    # GSE96058 usually has "overall survival days" and "vital status"
    # Or "os days" / "event"
    # Let's clean column names
    clin_df.columns = [c.lower().replace(' ', '_') for c in clin_df.columns]
    
    print(f"GSE96058 Clinical Columns: {clin_df.columns.tolist()}")
    
    # Map to time_to_event and event_status
    if 'overall_survival_days' in clin_df.columns:
        clin_df['time_to_event'] = pd.to_numeric(clin_df['overall_survival_days'], errors='coerce') / 30.0 # Months
    elif 'os_days' in clin_df.columns:
        clin_df['time_to_event'] = pd.to_numeric(clin_df['os_days'], errors='coerce') / 30.0
        
    if 'vital_status' in clin_df.columns:
        # "alive", "dead"
        clin_df['event_status'] = clin_df['vital_status'].apply(lambda x: 1 if 'dead' in str(x).lower() else 0)
    elif 'event' in clin_df.columns:
         clin_df['event_status'] = pd.to_numeric(clin_df['event'], errors='coerce')
    elif 'overall_survival_event' in clin_df.columns:
         clin_df['event_status'] = pd.to_numeric(clin_df['overall_survival_event'], errors='coerce')

    # Binary prediction target
    clin_df['high_risk'] = np.nan
    clin_df.loc[(clin_df['event_status'] == 1) & (clin_df['time_to_event'] < 60), 'high_risk'] = 1
    clin_df.loc[(clin_df['time_to_event'] >= 60), 'high_risk'] = 0
    clin_df = clin_df.dropna(subset=['high_risk'])
    clin_df['high_risk'] = clin_df['high_risk'].astype(int)
    
    print(f"GSE96058 Clinical (Filtered): {len(clin_df)} samples")

    # Load Expression
    # "Gene" "Sample1" "Sample2"...
    exp_df = pd.read_csv(exp_path, compression='gzip', index_col=0) # Index is Gene
    
    # Check if samples match
    exp_samples = exp_df.columns.tolist()
    # Map Geo Accession to Exp columns?
    # Usually exp columns are GSM IDs or similar.
    
    # Intersect
    common = list(set(clin_df['sample_id']) & set(exp_samples))
    print(f"Common samples (Est): {len(common)}")
    
    # Transpose expression to Samples x Genes
    exp_df = exp_df[common].T
    exp_df.index.name = 'sample_id'
    exp_df = exp_df.reset_index()
    
    clin_df = clin_df[clin_df['sample_id'].isin(common)]
    
    return exp_df, clin_df

def load_data(data_dir: str = "data", use_synthetic: bool = False, source: str = "TCGA") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads data. Source: "TCGA" or "GSE96058"
    """
    if use_synthetic:
        return generate_synthetic_data()
    
    try:
        if source == "TCGA":
            return load_real_tcga_data(data_dir)
        elif source == "GSE96058":
            return load_gse96058_data(data_dir)
        else:
            raise ValueError("Unknown source")
    except Exception as e:
        print(f"Failed to load real data ({source}): {e}")
        return generate_synthetic_data()

if __name__ == "__main__":
    # Test generation
    exp, clin = generate_synthetic_data(100, 50)
    print(f"Expression shape: {exp.shape}")
    print(f"Clinical shape: {clin.shape}")
    print(clin.head())
