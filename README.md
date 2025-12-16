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

python -m scripts.load_splice "src/data/molecular+biology+splice+junction+gene+sequences/splice.data" --out splice.parquet
python -m scripts.run_dbscan "src/data/molecular+biology+splice+junction+gene+sequences/splice.data" \
  --metric "weighted_angle_rho=0.5" --max-n 60 --n 400 --tune --trials 40
