# Predicting Breast Cancer Progression Risk using Gene Expression & AI

**Authors:** Suhaan Thayyil & Eshaan Nidee  
**Status:** Complete (ISEF 2026 Project)

## 📌 Project Overview
This project uses machine learning (Gradient Boosting) to predict whether a breast cancer patient will survive more than 5 years (Low Risk) or suffer rapid progression (High Risk). We utilized the TCGA-BRCA and SCAN-B datasets and engineered features based on biological pathways (Proliferation, Angiogenesis, etc.).

## 🚀 Key Results
- **Validation AUC:** 0.86 (Scan-B Cohort)
- **Top Predictors:** Ki67 (Proliferation), Age, Estrogen Response.
- **Explainability:** SHAP analysis confirmed that the model learned valid biological mechanisms, not just statistical noise.

## 📂 Repository Structure
- `notebooks/`: The complete Jupyter notebooks used to train the models.
- `data/`: Processed feature matrices (raw patient data excluded for privacy).
- `paper/`: The final research paper in LaTeX format.
- `figures/`: Generated plots and results.
- `src/`: Python source code for data loading and feature engineering.

## 🛠️ How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run `notebooks/01_model_training.ipynb`
