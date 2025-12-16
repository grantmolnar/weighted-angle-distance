# String Metric Experiments

This repo compares a custom **ρ-weighted angle distance on strings** against
standard string distances (from `textdistance`) for clustering tasks (e.g. DBSCAN
on genomic data).

## Setup

```bash
conda env create -f environment.yml
conda activate string-metric-experiments

# Install package in editable mode
pip install -e .

python -m scripts.run_dbscan --max-rows 400
pytest --cov=src/clustering/dbscan.py --cov-report=term-missing