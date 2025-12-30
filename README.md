## String Metric Experiments

This repository evaluates a custom **rho-weighted angle distance on strings** against
standard string distances (e.g. Levenshtein, Jaro–Winkler, LCS, k-gram variants) on
clustering tasks such as **DBSCAN** on sequence data.

The repo includes:
- A small clustering "suite" that runs importer × distance experiments and writes results + plots.
- Synthetic tandem-repeat data generation (useful for “stutter”-heavy sequences).
- Download/loader utilities for real STRSeq-style datasets.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate string-metric-experiments

# Install this repo as an editable package
pip install -e .

## Run Clustering Experiments

python -m scripts.run_dbscan

Outputs are written under outputs/dbscan_suite/ (CSV/Parquet + optional plots).

## Quality Checks

pytest --cov
black .
mypy . --check-untyped-defs

## Notes 

If you add new datasets, implement a loader that returns a Polars DataFrame with columns:

label (ground truth class label, string)
sequence (string to compare)

Pairwise distance computation is currently O(N^2) since we create the entire distance matrix; keep datasets small unless you optimize it.