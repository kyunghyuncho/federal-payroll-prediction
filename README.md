# Federal Payroll Prediction Demo

Educational Streamlit app for comparing regression losses on federal salary prediction using Nomic text embeddings, a year feature, and PyTorch Lightning.

## Environment

This project uses `uv` and a local virtual environment at `.venv`.

## Planned Dataset Shape

The app expects a parquet dataset with:

- `768` embedding columns
- `Year`
- `Target_Salary`

Recommended teaching subset sizes:

- `10,000-30,000` rows for fast classroom demos
- `50,000+` rows only if embeddings are already precomputed

## Quick Start

```bash
uv sync
.venv/bin/streamlit run app.py
```

## Data Acquisition

The repo includes a helper script for early data inspection and development subsetting:

```bash
.venv/bin/python scripts/fetch_and_profile_data.py --help
```

The script can:

- download a remote file or inspect a local one
- read CSV, TSV, parquet, JSONL, pickle, and ZIP-wrapped tabular files
- write a JSON profile with row count, columns, dtypes, and sample records
- save a reproducible random parquet subset for faster development
- choose a development subset size automatically from the total row count

Example:

```bash
.venv/bin/python scripts/fetch_and_profile_data.py \
  --input-path data/raw/your_dataset.parquet \
  --auto-subset
```

Auto-subset defaults:

- `<= 30,000` rows: keep the full dataset
- `30,001-150,000` rows: save a `20,000` row subset
- `> 150,000` rows: save a `30,000` row subset

You can still override this with `--subset-size`.

## Notes

- Keep embeddings precomputed so students are not blocked on preprocessing.
- A smaller ready-to-use parquet file is recommended for class, with a larger optional extension dataset for homework or exploration.
