# A Biologically Informed Explainable Machine Learning Framework for Breast Cancer Progression Risk Stratification From Tumor Gene Expression

**Authors:** Suhaan Thayyil, Eshaan Nidee

## Abstract

We present a biologically informed machine learning framework for predicting breast cancer progression risk using pathway-level gene expression features. Seven curated biological pathways (45 genes) are distilled into interpretable scores via mean z-score aggregation and combined with clinical variables. We train Elastic Net, Random Forest, and Gradient Boosting classifiers, complemented by Cox proportional hazards survival models and SHAP-based explainability analysis. The framework is trained on TCGA-BRCA (n=213) and externally validated on the independent SCAN-B/GSE96058 cohort (n=1,483), achieving a combined-feature AUC of 0.856 and Cox C-index of 0.827. An ablation study quantifies the contribution of pathway vs. clinical features, ssGSEA baseline comparison validates the scoring methodology, and cross-fold stability analysis (mean Spearman rho=0.820) confirms robust feature importance rankings.

**Preprint:** bioRxiv [DOI pending]

## Repository Structure

```
breast-cancer-prediction/
├── README.md
├── LICENSE                              MIT License
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   ├── README.md                    Download instructions
│   │   └── download_data.sh             Downloads GSE96058 expression data
│   ├── processed/
│   │   ├── 02_tcga_feature_matrix.csv   TCGA pathway scores (213 patients)
│   │   └── ablation_results.csv         Ablation study results
│   └── clinical/
│       ├── 01_gse96058_clinical.csv     SCAN-B clinical data (1,483 patients)
│       └── 01_tcga_clinical.csv         TCGA clinical data
├── notebooks/
│   ├── 01_data_preprocessing.ipynb
│   ├── 02_pathway_score_computation.ipynb
│   ├── 03_model_training_tcga.ipynb
│   ├── 04_model_training_gse96058.ipynb
│   ├── 05_ablation_study.ipynb
│   ├── 06_survival_analysis.ipynb
│   ├── 07_ssgsea_baseline.ipynb
│   ├── 08_shap_analysis.ipynb
│   ├── 09_stability_analysis.ipynb
│   ├── 10_calibration.ipynb
│   ├── 11_figure_generation.ipynb
│   ├── 12_subtype_baseline.ipynb
│   └── 13_additional_analyses.ipynb       # Peer-review analyses (PH tests, NRI, DCA)
├── src/
│   ├── __init__.py
│   ├── data_loader.py                   Data loading utilities
│   ├── preprocessing.py                 Z-score normalization, clinical encoding
│   ├── features.py                      Pathway definitions and scoring
│   ├── models.py                        Classifiers and CV evaluation
│   ├── survival.py                      Cox PH and Kaplan-Meier analysis
│   └── baselines.py                     ssGSEA baseline implementation
├── figures/
│   ├── fig_ablation.png
│   ├── fig_calibration.png
│   ├── fig_comparison.png
│   ├── fig_kaplan_meier.png
│   ├── fig_roc.png
│   ├── fig_shap_beeswarm.png
│   ├── fig_shap_pathway_only.png
│   ├── fig_stability.png
│   ├── fig_waterfall_high.png
│   └── fig_waterfall_low.png
├── results/
│   ├── 03_model_performance.csv
│   ├── 04_elastic_net_coefficients.csv
│   ├── 04_shap_values_all_patients.csv
│   ├── ablation_results.csv
│   ├── schoenfeld_results.csv              # PH assumption tests
│   ├── permutation_importance.csv          # Permutation importance rankings
│   ├── bootstrap_ci_results.csv            # Bootstrap 95% CIs
│   ├── treatment_stratified_results.csv    # Treatment subgroup analysis
│   ├── dca_nri_results.csv                 # Decision curve + NRI metrics
│   └── sensitivity_tumor_size_ph.csv       # Time-dependent tumor size sensitivity
└── paper/
    ├── paper.tex                        # bioRxiv/Zenodo preprint
    ├── ieee_jbhi_paper.tex            # IEEE JBHI submission
    └── mlwa_paper.tex                   # Machine Learning with Applications (Elsevier)
```

## Reproduction Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/suhaanthayyil/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download expression data** (~565 MB download, ~1.8 GB decompressed)
   ```bash
   cd data/raw && bash download_data.sh && cd ../..
   ```

4. **Run notebooks in order**
   ```bash
   jupyter notebook
   ```
   Execute notebooks 01 through 11 sequentially. Each notebook:
   - Imports reusable functions from `src/`
   - Loads data from `data/`
   - Prints key results for verification
   - Saves outputs to `results/` or `figures/`

All figures are regenerated in `figures/` and all result CSVs in `results/`.

## Key Results

| Analysis | Metric | Pathway Only | Clinical Only | Combined |
|----------|--------|-------------|---------------|----------|
| Binary Classification (GSE96058) | Best AUC | 0.645 (RF) | 0.851 (GB) | 0.856 (RF) |
| Cox Survival (GSE96058) | C-index | 0.626 +/- 0.030 | 0.822 +/- 0.030 | 0.827 +/- 0.031 |
| ssGSEA Comparison | AUC Delta | +0.010 to +0.038 | - | - |
| Stability | Spearman rho | - | - | 0.820 |
| KM Log-rank | p-value | - | - | < 5.7e-53 |
| Subtype-Only Baseline | AUC | - | - | 0.613 |
| Combined vs Subtype | Delta | - | - | +0.243 |
| Within Luminal A | CV C-index | - | - | 0.848 |
| Within Luminal A | Log-rank p | - | - | 5.86e-22 |

## Additional analyses (peer-review ready)

These analyses were added in response to anticipated reviewer concerns and are
documented in `notebooks/13_additional_analyses.ipynb`.

| Analysis | Result | Purpose |
|----------|--------|---------|
| Schoenfeld residual tests | 8/9 covariates pass PH at α=0.05; tumor size mild violation (p=0.007) | Validates Cox model assumption |
| Sensitivity (time-dependent tumor size) | ΔC-index = −0.001 | Confirms PH violation does not affect results |
| Permutation importance | Ki-67 and age rank 1 and 2 (matches SHAP) | Validates SHAP rankings with independent method |
| Treatment-stratified Cox | All 4 subgroups p < 1e-4 | Confirms risk signal not explained by treatment confounding |
| Decision curve analysis | Positive net benefit, thresholds 0.04–0.50 | Demonstrates clinical utility beyond AUC |
| Net reclassification index | NRI = 0.801 (95% CI: 0.679–0.911) vs subtype baseline | Quantifies improvement in patient classification |
| Bootstrap 95% CIs | Added to all main AUC and C-index values | Statistical rigor per reviewer standards |

## Journal papers

| Version | File | Target journal | Status |
|---------|------|----------------|--------|
| Preprint | paper/paper.tex | Zenodo / bioRxiv | Posted — DOI: 10.5281/zenodo.18749909 |
| IEEE version | paper/ieee_jbhi_paper.tex | IEEE JBHI (IF 6.7) | Ready to submit with professor co-author |
| MLWA version | paper/mlwa_paper.tex | Machine Learning with Applications (IF 4.9) | Ready to submit |

## Citation

```bibtex
@misc{thayyil2026breast,
  author    = {Thayyil, Suhaan and Nidee, Eshaan},
  title     = {A Biologically Informed Explainable Machine Learning Framework
               for Breast Cancer Progression Risk Stratification from
               Tumor Gene Expression},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.18749909},
  url       = {https://doi.org/10.5281/zenodo.18749909}
}
```

## Data Sources

- **TCGA-BRCA**: [GDC Data Portal](https://portal.gdc.cancer.gov) - Breast Invasive Carcinoma, RNA-seq (HTSeq-FPKM)
- **GSE96058 / SCAN-B**: [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058) - 3,273 breast cancer samples with RNA-seq and clinical data



## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
