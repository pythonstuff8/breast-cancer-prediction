#!/bin/bash
# Download raw gene expression data for the breast cancer prediction project.
# Run this script from the data/raw/ directory.

set -e

echo "=== Downloading GSE96058 gene expression data (~565MB compressed) ==="
curl -L -o GSE96058_gene_expression.csv.gz \
  "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96058/suppl/GSE96058_gene_expression_3273_samples_and_136_replicates_transformed.csv.gz"

echo "=== Decompressing (~1.8GB decompressed) ==="
gunzip GSE96058_gene_expression.csv.gz

echo "=== Done ==="
echo "File: GSE96058_gene_expression.csv"
echo "Size: $(du -h GSE96058_gene_expression.csv | cut -f1)"
