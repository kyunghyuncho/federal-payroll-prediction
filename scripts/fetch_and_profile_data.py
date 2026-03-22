#!/usr/bin/env python3
"""Download and profile a tabular dataset for the salary demo.

This script is built to be practical during early project setup:
- download from a URL or inspect an existing local file
- handle plain files and ZIP archives
- read CSV, TSV, parquet, JSONL, and pickle tabular data
- optionally create a reproducible random subset for faster development
- emit a JSON profile with row count, columns, dtypes, and simple statistics

Example:
    .venv/bin/python scripts/fetch_and_profile_data.py \
        --url https://example.com/data.zip \
        --member employment.csv \
        --subset-size 10000
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pandas as pd


DEFAULT_URLS = {
    "FedScope Employment Data File 1 (March 2025)": "https://www.opm.gov/data/datasets/Files/756/a1acc4f3-0c10-45e3-ac1f-0ee7f5769e1d.zip",
    "FedScope Employment Data File 2 (March 2025)": "https://www.opm.gov/data/datasets/Files/758/18693acd-e0d7-4cb5-b15b-bd455a3a432c.zip",
    "FedScope Employment Data File 3 (March 2025)": "https://www.opm.gov/data/datasets/Files/759/3c93cbe4-ae79-4881-8562-5892df28744d.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, subset, and profile a dataset for development.",
    )
    parser.add_argument(
        "--url",
        help="Remote dataset URL. Can point to CSV, parquet, pickle, JSONL, or ZIP.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Existing local file to inspect instead of downloading.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory for downloaded files and generated profiles.",
    )
    parser.add_argument(
        "--download-name",
        help="Optional filename override for the downloaded artifact.",
    )
    parser.add_argument(
        "--member",
        help="ZIP member to read. If omitted, the first tabular member is used.",
    )
    parser.add_argument(
        "--format",
        choices=["auto", "csv", "tsv", "parquet", "jsonl", "pickle"],
        default="auto",
        help="Tabular format for the selected file or ZIP member.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Text encoding for CSV, TSV, and JSONL files.",
    )
    parser.add_argument(
        "--sep",
        help="Delimiter override for CSV-style inputs.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Random subset size to save for development. Overrides auto sizing. Use 0 to skip subset creation.",
    )
    parser.add_argument(
        "--auto-subset",
        action="store_true",
        help="Choose a development subset size automatically from the full row count.",
    )
    parser.add_argument(
        "--small-threshold",
        type=int,
        default=30000,
        help="Keep the full dataset when row count is at or below this threshold.",
    )
    parser.add_argument(
        "--medium-threshold",
        type=int,
        default=150000,
        help="Use the medium subset rule when row count is at or below this threshold.",
    )
    parser.add_argument(
        "--medium-subset-size",
        type=int,
        default=20000,
        help="Subset size for medium datasets.",
    )
    parser.add_argument(
        "--large-subset-size",
        type=int,
        default=30000,
        help="Subset size for large datasets.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsetting.",
    )
    parser.add_argument(
        "--profile-head",
        type=int,
        default=5,
        help="Number of example records to store in the JSON profile.",
    )
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def infer_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    candidate = Path(parsed.path).name
    return candidate or "downloaded_dataset"


def download_file(url: str, output_path: Path) -> Path:
    ensure_parent(output_path)
    with urlopen(url) as response, output_path.open("wb") as fh:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
    return output_path


def is_tabular_name(name: str) -> bool:
    suffix = Path(name).suffix.lower()
    return suffix in {".csv", ".tsv", ".txt", ".parquet", ".jsonl", ".pkl", ".pickle"}


def list_zip_members(input_path: Path) -> list[str]:
    if not zipfile.is_zipfile(input_path):
        return []
    with zipfile.ZipFile(input_path) as zf:
        return [name for name in zf.namelist() if not name.endswith("/")]


def choose_zip_member(zf: zipfile.ZipFile, requested_member: str | None) -> str:
    names = [name for name in zf.namelist() if not name.endswith("/")]
    if requested_member:
        if requested_member not in names:
            raise FileNotFoundError(f"ZIP member not found: {requested_member}")
        return requested_member

    tabular_names = [name for name in names if is_tabular_name(name)]
    if not tabular_names:
        raise ValueError("No tabular file found inside ZIP archive.")
    return tabular_names[0]


def infer_format(path_like: str, explicit_format: str) -> str:
    if explicit_format != "auto":
        return explicit_format

    suffix = Path(path_like).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".txt":
        return "txt"
    if suffix == ".parquet":
        return "parquet"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix in {".pkl", ".pickle"}:
        return "pickle"
    raise ValueError(f"Could not infer file format from: {path_like}")


def read_dataframe_from_bytes(
    raw_bytes: bytes,
    logical_name: str,
    fmt: str,
    encoding: str,
    sep: str | None,
) -> pd.DataFrame:
    detected = infer_format(logical_name, fmt)

    if detected == "csv":
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=sep or ",")
    if detected == "tsv":
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=sep or "\t")
    if detected == "txt":
        text_sep = sep or "|"
        return pd.read_csv(io.BytesIO(raw_bytes), encoding=encoding, sep=text_sep)
    if detected == "parquet":
        return pd.read_parquet(io.BytesIO(raw_bytes))
    if detected == "jsonl":
        return pd.read_json(io.BytesIO(raw_bytes), lines=True)
    if detected == "pickle":
        obj = pickle.loads(raw_bytes)
        if isinstance(obj, pd.DataFrame):
            return obj
        raise TypeError("Pickle file did not contain a pandas DataFrame.")

    raise ValueError(f"Unsupported format: {detected}")


def load_dataframe(
    input_path: Path,
    fmt: str,
    member: str | None,
    encoding: str,
    sep: str | None,
) -> tuple[pd.DataFrame, str]:
    if zipfile.is_zipfile(input_path):
        with zipfile.ZipFile(input_path) as zf:
            chosen_member = choose_zip_member(zf, member)
            raw_bytes = zf.read(chosen_member)
        df = read_dataframe_from_bytes(raw_bytes, chosen_member, fmt, encoding, sep)
        return df, chosen_member

    raw_bytes = input_path.read_bytes()
    df = read_dataframe_from_bytes(raw_bytes, input_path.name, fmt, encoding, sep)
    return df, input_path.name


def make_subset(df: pd.DataFrame, subset_size: int, seed: int) -> pd.DataFrame:
    if subset_size <= 0 or subset_size >= len(df):
        return df.copy()
    return df.sample(n=subset_size, random_state=seed).reset_index(drop=True)


def choose_subset_size(args: argparse.Namespace, row_count: int) -> int:
    if args.subset_size is not None:
        return max(0, args.subset_size)

    if not args.auto_subset:
        return 0

    if row_count <= args.small_threshold:
        return row_count
    if row_count <= args.medium_threshold:
        return min(args.medium_subset_size, row_count)
    return min(args.large_subset_size, row_count)


def safe_number(value: object) -> object:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def summarize_dataframe(df: pd.DataFrame, sample_rows: int) -> dict[str, object]:
    numeric = df.select_dtypes(include=["number"])

    column_preview = []
    for column in df.columns[:25]:
        preview = {
            "name": column,
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(df[column]):
            preview["min"] = safe_number(df[column].min())
            preview["max"] = safe_number(df[column].max())
            preview["mean"] = safe_number(df[column].mean())
        column_preview.append(preview)

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": [str(col) for col in df.columns.tolist()],
        "dtypes": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_columns": [str(col) for col in numeric.columns.tolist()],
        "column_preview": column_preview,
        "head": df.head(sample_rows).to_dict(orient="records"),
    }


def write_profile(output_path: Path, payload: dict[str, object]) -> None:
    ensure_parent(output_path)
    output_path.write_text(json.dumps(payload, indent=2, default=str))


def save_subset(subset_df: pd.DataFrame, output_path: Path) -> None:
    ensure_parent(output_path)
    subset_df.to_parquet(output_path, index=False)


def print_summary(profile: dict[str, object]) -> None:
    print(f"Rows: {profile['rows']}")
    print(f"Columns: {profile['columns']}")
    print("First columns:")
    for name in profile["column_names"][:10]:
        print(f"  - {name}")


def main() -> int:
    args = parse_args()

    if not args.url and not args.input_path:
        print("Provide either --url or --input-path.", file=sys.stderr)
        print(
            "Official OPM dataset catalog: https://www.opm.gov/data/datasets/",
            file=sys.stderr,
        )
        return 2

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.input_path:
        input_path = args.input_path
    else:
        download_name = args.download_name or infer_name_from_url(args.url)
        input_path = output_dir / download_name
        if not input_path.exists():
            print(f"Downloading {args.url} -> {input_path}")
            download_file(args.url, input_path)
        else:
            print(f"Reusing existing download: {input_path}")

    print(f"Loading dataset from {input_path}")
    df, logical_name = load_dataframe(
        input_path=input_path,
        fmt=args.format,
        member=args.member,
        encoding=args.encoding,
        sep=args.sep,
    )

    profile = summarize_dataframe(df, sample_rows=args.profile_head)
    profile["source_file"] = str(input_path)
    profile["logical_name"] = logical_name
    subset_size = choose_subset_size(args, len(df))
    profile["recommended_subset_size"] = subset_size

    profile_path = output_dir / f"{Path(logical_name).stem}_profile.json"
    write_profile(profile_path, profile)
    print_summary(profile)
    print(f"Recommended subset size: {subset_size}")
    print(f"Profile written to {profile_path}")

    if subset_size > 0:
        subset_df = make_subset(df, subset_size, args.seed)
        subset_path = output_dir / f"{Path(logical_name).stem}_subset_{len(subset_df)}.parquet"
        save_subset(subset_df, subset_path)
        print(f"Subset written to {subset_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
