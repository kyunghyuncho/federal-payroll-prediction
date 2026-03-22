#!/usr/bin/env python3
"""Download, validate, and subset OPM FedScope employment files.

This script is specific to the OPM FedScope Employment Data File 1/2/3 ZIPs.
Those archives contain a pipe-delimited `.txt` file and a PDF dictionary, so
they need slightly different handling than ordinary CSV/parquet inputs.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.fetch_and_profile_data import (
    DEFAULT_URLS,
    download_file,
    infer_name_from_url,
    list_zip_members,
    make_subset,
    save_subset,
    write_profile,
)


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and process an OPM FedScope employment ZIP.",
    )
    parser.add_argument(
        "--dataset",
        choices=list(DEFAULT_URLS.keys()),
        help="Built-in OPM dataset preset.",
    )
    parser.add_argument(
        "--url",
        help="Direct OPM ZIP URL. Overrides --dataset if provided.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        help="Existing local ZIP path instead of downloading.",
    )
    parser.add_argument(
        "--member",
        help="ZIP member to load. Defaults to the first .txt file.",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=10000,
        help="Row count for the saved development subset. Use 0 to skip.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subsetting.",
    )
    parser.add_argument(
        "--output-stem",
        default="opm_fedscope_employment",
        help="Base name for processed outputs.",
    )
    return parser.parse_args()


def resolve_source(args: argparse.Namespace) -> tuple[Path, str]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.input_path:
        return args.input_path, "local"

    url = args.url or DEFAULT_URLS.get(args.dataset or "", "")
    if not url:
        raise ValueError("Provide --input-path, --url, or --dataset.")

    destination = RAW_DIR / infer_name_from_url(url)
    if not destination.exists():
        download_file(url, destination)
    return destination, url


def choose_txt_member(input_path: Path, requested_member: str | None) -> str:
    members = list_zip_members(input_path)
    if requested_member:
        if requested_member not in members:
            raise FileNotFoundError(f"ZIP member not found: {requested_member}")
        return requested_member

    txt_members = [member for member in members if member.lower().endswith(".txt")]
    if not txt_members:
        raise ValueError("No .txt member found in the OPM ZIP archive.")
    return txt_members[0]


def load_opm_dataframe(input_path: Path, member: str) -> pd.DataFrame:
    with zipfile.ZipFile(input_path) as zf:
        with zf.open(member) as fh:
            df = pd.read_csv(fh, sep="|", quotechar='"')
    return df


def clean_opm_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    for column in ["COUNT", "DATECODE", "SALARY", "GRD"]:
        if column in cleaned.columns:
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if "DATECODE" in cleaned.columns:
        cleaned["Year"] = (cleaned["DATECODE"] // 100).astype("Int64")
    cleaned["salary_redacted"] = cleaned.get("SALARY").isna() if "SALARY" in cleaned.columns else True
    return cleaned


def summarize_opm(df: pd.DataFrame, input_path: Path, member: str) -> dict[str, object]:
    salary_series = df["SALARY"] if "SALARY" in df.columns else pd.Series(dtype=float)
    count_series = pd.to_numeric(df["COUNT"], errors="coerce") if "COUNT" in df.columns else pd.Series(dtype=float)
    return {
        "source_file": str(input_path),
        "zip_member": member,
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "usable_salary_rows": int(salary_series.notna().sum()) if not salary_series.empty else 0,
        "redacted_salary_rows": int(salary_series.isna().sum()) if not salary_series.empty else int(len(df)),
        "salary_is_usable": bool(salary_series.notna().any()) if not salary_series.empty else False,
        "count_sum": int(count_series.sum()) if not count_series.empty else None,
        "year_min": int(df["Year"].dropna().min()) if "Year" in df.columns and df["Year"].notna().any() else None,
        "year_max": int(df["Year"].dropna().max()) if "Year" in df.columns and df["Year"].notna().any() else None,
    }


def main() -> int:
    args = parse_args()
    input_path, source_label = resolve_source(args)
    member = choose_txt_member(input_path, args.member)
    df = clean_opm_dataframe(load_opm_dataframe(input_path, member))
    summary = summarize_opm(df, input_path, member)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full_output = PROCESSED_DIR / f"{args.output_stem}_full.parquet"
    profile_output = PROCESSED_DIR / f"{args.output_stem}_profile.json"
    save_subset(df, full_output)
    write_profile(profile_output, summary)

    print(json.dumps(summary, indent=2))
    print(f"Saved full parsed parquet to {full_output}")
    print(f"Saved profile to {profile_output}")

    if args.subset_size > 0:
        subset_df = make_subset(df, args.subset_size, args.seed)
        subset_output = PROCESSED_DIR / f"{args.output_stem}_subset_{len(subset_df)}.parquet"
        save_subset(subset_df, subset_output)
        print(f"Saved subset parquet to {subset_output}")

    if not summary["salary_is_usable"]:
        print(
            "WARNING: This OPM file does not contain usable salary values after parsing. "
            "It is not suitable as the training target for salary prediction."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
