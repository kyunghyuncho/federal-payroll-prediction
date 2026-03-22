from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from scripts.fetch_and_profile_data import (
    DEFAULT_URLS,
    download_file,
    infer_name_from_url,
    list_zip_members,
    load_dataframe,
    make_subset,
    save_subset,
    summarize_dataframe,
    write_profile,
)


DATA_DIR = Path("data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DEFAULT_DATA_PATH = PROCESSED_DATA_DIR / "federal_salary_embeddings.parquet"


def ensure_data_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(uploaded_file) -> Path:
    destination = RAW_DATA_DIR / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def choose_default_zip_member(members: list[str]) -> str | None:
    tabular_members = [name for name in members if Path(name).suffix.lower() in {".csv", ".tsv", ".parquet", ".jsonl", ".pkl", ".pickle"}]
    if not tabular_members:
        return members[0] if members else None
    preferred_names = ("employment", "datafile", "fedscope")
    for pattern in preferred_names:
        for member in tabular_members:
            if pattern in member.lower():
                return member
    return tabular_members[0]


def choose_subset_size(
    row_count: int,
    use_auto_subset: bool,
    explicit_subset_size: int,
    small_threshold: int,
    medium_threshold: int,
    medium_subset_size: int,
    large_subset_size: int,
) -> int:
    if explicit_subset_size > 0:
        return min(explicit_subset_size, row_count)
    if not use_auto_subset:
        return row_count
    if row_count <= small_threshold:
        return row_count
    if row_count <= medium_threshold:
        return min(medium_subset_size, row_count)
    return min(large_subset_size, row_count)


def render_profile(profile: dict[str, object], df: pd.DataFrame) -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Rows", f"{profile['rows']:,}")
    metric_cols[1].metric("Columns", f"{profile['columns']:,}")
    metric_cols[2].metric("Numeric Columns", f"{len(profile['numeric_columns']):,}")
    metric_cols[3].metric(
        "Missing Cells",
        f"{int(df.isna().sum().sum()):,}",
    )

    st.subheader("Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Column Summary")
    st.dataframe(pd.DataFrame(profile["column_preview"]), use_container_width=True)

    numeric_columns = list(profile["numeric_columns"])
    if numeric_columns:
        st.subheader("Numeric Distribution")
        chart_col, details_col = st.columns([2, 1])
        default_column = "Target_Salary" if "Target_Salary" in numeric_columns else numeric_columns[0]
        selected_numeric = chart_col.selectbox(
            "Numeric column",
            numeric_columns,
            index=numeric_columns.index(default_column),
        )
        bins = details_col.slider("Histogram bins", min_value=10, max_value=100, value=40)
        fig = px.histogram(
            df,
            x=selected_numeric,
            nbins=bins,
            title=f"Distribution of {selected_numeric}",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "Year" in df.columns and pd.api.types.is_numeric_dtype(df["Year"]):
        st.subheader("Records by Year")
        year_counts = (
            df["Year"]
            .dropna()
            .astype(int)
            .value_counts()
            .sort_index()
            .rename_axis("Year")
            .reset_index(name="Count")
        )
        year_fig = px.bar(year_counts, x="Year", y="Count", title="Dataset coverage by year")
        st.plotly_chart(year_fig, use_container_width=True)

    if "Target_Salary" in df.columns and pd.api.types.is_numeric_dtype(df["Target_Salary"]):
        st.subheader("Salary by Year")
        if "Year" in df.columns and pd.api.types.is_numeric_dtype(df["Year"]):
            salary_by_year = (
                df[["Year", "Target_Salary"]]
                .dropna()
                .assign(Year=lambda frame: frame["Year"].astype(int))
                .groupby("Year", as_index=False)["Target_Salary"]
                .median()
            )
            line_fig = px.line(
                salary_by_year,
                x="Year",
                y="Target_Salary",
                markers=True,
                title="Median salary by year",
            )
            st.plotly_chart(line_fig, use_container_width=True)


def process_dataset(
    input_path: Path,
    fmt: str,
    member: str | None,
    encoding: str,
    sep: str | None,
    use_auto_subset: bool,
    explicit_subset_size: int,
    small_threshold: int,
    medium_threshold: int,
    medium_subset_size: int,
    large_subset_size: int,
    seed: int,
    profile_head: int,
    processed_name: str,
) -> tuple[pd.DataFrame, dict[str, object], Path]:
    df, logical_name = load_dataframe(
        input_path=input_path,
        fmt=fmt,
        member=member,
        encoding=encoding,
        sep=sep,
    )
    subset_size = choose_subset_size(
        row_count=len(df),
        use_auto_subset=use_auto_subset,
        explicit_subset_size=explicit_subset_size,
        small_threshold=small_threshold,
        medium_threshold=medium_threshold,
        medium_subset_size=medium_subset_size,
        large_subset_size=large_subset_size,
    )
    subset_df = make_subset(df, subset_size, seed).reset_index(drop=True)
    profile = summarize_dataframe(subset_df, sample_rows=profile_head)
    profile["source_file"] = str(input_path)
    profile["logical_name"] = logical_name
    profile["full_row_count"] = int(len(df))
    profile["subset_row_count"] = int(len(subset_df))
    profile["recommended_subset_size"] = int(subset_size)

    processed_path = PROCESSED_DATA_DIR / processed_name
    save_subset(subset_df, processed_path)
    profile_path = PROCESSED_DATA_DIR / f"{processed_path.stem}_profile.json"
    write_profile(profile_path, profile)
    return subset_df, profile, processed_path


def render_data_page() -> None:
    st.title("Federal Salary Data Prep")
    st.write(
        "Download or load a dataset, create a development-sized subset, "
        "and inspect the data before training the regression model."
    )

    source_tab, analyze_tab = st.tabs(["Prepare Data", "Analyze Latest Processed Data"])

    with source_tab:
        with st.sidebar:
            st.header("Processing Options")
            source_type = st.radio(
                "Source",
                options=["Local path", "Upload file", "URL download"],
            )
            file_format = st.selectbox(
                "Format",
                options=["auto", "csv", "tsv", "txt", "parquet", "jsonl", "pickle"],
                index=0,
            )
            encoding = st.text_input("Text encoding", value="utf-8")
            separator = st.text_input("Delimiter override (optional)")
            processed_name = st.text_input(
                "Processed parquet name",
                value=DEFAULT_DATA_PATH.name,
            )

            st.header("Subset Controls")
            subset_mode = st.radio(
                "Subset mode",
                options=["Auto subset", "Fixed size", "Keep all rows"],
            )
            explicit_subset_size = 0
            use_auto_subset = subset_mode == "Auto subset"
            if subset_mode == "Fixed size":
                explicit_subset_size = st.number_input(
                    "Subset size",
                    min_value=1,
                    value=10000,
                    step=1000,
                )
            small_threshold = st.number_input("Small threshold", min_value=1, value=30000, step=1000)
            medium_threshold = st.number_input("Medium threshold", min_value=1, value=150000, step=5000)
            medium_subset_size = st.number_input(
                "Medium subset size",
                min_value=1,
                value=20000,
                step=1000,
            )
            large_subset_size = st.number_input(
                "Large subset size",
                min_value=1,
                value=30000,
                step=1000,
            )
            seed = st.number_input("Random seed", min_value=0, value=42, step=1)
            profile_head = st.number_input("Profile head rows", min_value=1, value=5, step=1)

        local_path = ""
        uploaded_file = None
        download_url = ""
        download_name = ""
        zip_member = ""
        candidate_zip_path: Path | None = None

        if source_type == "Local path":
            local_path = st.text_input(
                "Local dataset path",
                value=str(DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else RAW_DATA_DIR / "your_dataset.parquet"),
            )
            path_candidate = Path(local_path).expanduser()
            if path_candidate.exists():
                candidate_zip_path = path_candidate
        elif source_type == "Upload file":
            uploaded_file = st.file_uploader(
                "Upload dataset",
                type=["csv", "tsv", "parquet", "jsonl", "pkl", "pickle", "zip"],
            )
            if uploaded_file is not None and uploaded_file.name.lower().endswith(".zip"):
                temp_preview_path = RAW_DATA_DIR / uploaded_file.name
                temp_preview_path.write_bytes(uploaded_file.getbuffer())
                candidate_zip_path = temp_preview_path
        else:
            url_choice = st.selectbox(
                "Dataset preset",
                options=["Custom URL", *DEFAULT_URLS.keys()],
            )
            default_url = DEFAULT_URLS.get(url_choice, "")
            download_url = st.text_input("Dataset URL", value=default_url)
            suggested_name = infer_name_from_url(download_url) if download_url else ""
            download_name = st.text_input("Saved download name (optional)", value=suggested_name)
            if download_url.strip():
                expected_name = download_name.strip() or infer_name_from_url(download_url)
                possible_path = RAW_DATA_DIR / expected_name
                if possible_path.exists():
                    candidate_zip_path = possible_path

        if candidate_zip_path and candidate_zip_path.exists() and zipfile.is_zipfile(candidate_zip_path):
            zip_members = list_zip_members(candidate_zip_path)
            default_member = choose_default_zip_member(zip_members)
            if zip_members:
                zip_member = st.selectbox(
                    "ZIP member",
                    options=zip_members,
                    index=zip_members.index(default_member) if default_member in zip_members else 0,
                    help="Choose which file inside the ZIP archive should be loaded.",
                )
                with st.expander("ZIP contents"):
                    st.write("\n".join(zip_members))
        else:
            zip_member = st.text_input("ZIP member (optional)")

        if st.button("Process Dataset", type="primary"):
            try:
                ensure_data_dirs()
                if source_type == "Local path":
                    input_path = Path(local_path).expanduser()
                    if not input_path.exists():
                        raise FileNotFoundError(f"Dataset path does not exist: {input_path}")
                elif source_type == "Upload file":
                    if uploaded_file is None:
                        raise ValueError("Upload a file before processing.")
                    input_path = save_uploaded_file(uploaded_file)
                else:
                    if not download_url.strip():
                        raise ValueError("Provide a dataset URL before processing.")
                    resolved_name = download_name.strip() or infer_name_from_url(download_url)
                    input_path = RAW_DATA_DIR / resolved_name
                    with st.spinner("Downloading dataset..."):
                        download_file(download_url, input_path)

                with st.spinner("Loading, subsetting, and profiling dataset..."):
                    subset_df, profile, processed_path = process_dataset(
                        input_path=input_path,
                        fmt=file_format,
                        member=zip_member.strip() or None,
                        encoding=encoding,
                        sep=separator or None,
                        use_auto_subset=use_auto_subset,
                        explicit_subset_size=int(explicit_subset_size),
                        small_threshold=int(small_threshold),
                        medium_threshold=int(medium_threshold),
                        medium_subset_size=int(medium_subset_size),
                        large_subset_size=int(large_subset_size),
                        seed=int(seed),
                        profile_head=int(profile_head),
                        processed_name=processed_name,
                    )
                st.session_state["latest_processed_df"] = subset_df
                st.session_state["latest_profile"] = profile
                st.session_state["latest_processed_path"] = str(processed_path)
                st.success(f"Processed dataset saved to {processed_path}")
            except Exception as exc:
                st.error(str(exc))

        if "latest_profile" in st.session_state:
            st.divider()
            st.subheader("Latest Processing Result")
            render_profile(
                st.session_state["latest_profile"],
                st.session_state["latest_processed_df"],
            )
            latest_processed_path = Path(st.session_state["latest_processed_path"])
            st.download_button(
                "Download latest processed parquet",
                data=latest_processed_path.read_bytes(),
                file_name=latest_processed_path.name,
                mime="application/octet-stream",
            )
            st.download_button(
                "Download latest profile JSON",
                data=json.dumps(st.session_state["latest_profile"], indent=2, default=str),
                file_name="dataset_profile.json",
                mime="application/json",
            )

    with analyze_tab:
        latest_path = st.session_state.get("latest_processed_path", str(DEFAULT_DATA_PATH))
        analysis_path = st.text_input("Processed parquet path", value=latest_path)

        if st.button("Load Processed Dataset"):
            try:
                df = pd.read_parquet(analysis_path)
                profile = summarize_dataframe(df, sample_rows=5)
                st.session_state["latest_processed_df"] = df
                st.session_state["latest_profile"] = profile
                st.session_state["latest_processed_path"] = analysis_path
            except Exception as exc:
                st.error(str(exc))

        if "latest_profile" in st.session_state:
            render_profile(
                st.session_state["latest_profile"],
                st.session_state["latest_processed_df"],
            )
        else:
            st.info("Process or load a dataset to see the analysis dashboard.")


def main() -> None:
    st.set_page_config(page_title="Federal Salary Prediction Demo", layout="wide")
    ensure_data_dirs()

    page = st.sidebar.radio(
        "Page",
        options=["Data Prep", "Training Demo"],
    )

    if page == "Data Prep":
        render_data_page()
        return

    st.title("Federal Salary Prediction Demo")
    st.write(
        "The training demo will compare MSE, MAE, and quantile regression "
        "after the dataset pipeline is prepared."
    )
    if DEFAULT_DATA_PATH.exists():
        st.success(f"Processed dataset found at {DEFAULT_DATA_PATH}")
    else:
        st.warning(
            "No processed dataset found yet. Use the Data Prep page to create one."
        )


if __name__ == "__main__":
    main()
