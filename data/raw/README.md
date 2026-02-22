# Raw Data

Raw data files are not stored in this repository due to their size. Use the instructions below to obtain them.

## GSE96058 / SCAN-B Gene Expression

**Source:** [GEO Accession GSE96058](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE96058)

Download automatically:
```bash
bash download_data.sh
```

Or manually download from:
```
https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz
```

- Compressed: ~565 MB
- Decompressed: ~1.8 GB
- Format: Genes as rows, samples (F1, F2, ...) as columns, log2-transformed expression values

## TCGA-BRCA

**Source:** [GDC Data Portal](https://portal.gdc.cancer.gov)

- Project: TCGA-BRCA (Breast Invasive Carcinoma)
- Data type: RNA-seq gene expression (HTSeq-FPKM, log2 RSEM normalized)
- The preprocessed TCGA feature matrix is provided in `data/processed/02_tcga_feature_matrix.csv`

## Clinical Data

Clinical metadata for both cohorts is provided in `data/clinical/` and is small enough to include in the repository.
