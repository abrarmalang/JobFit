"""
Data Consolidation

Consolidates raw job data from multiple sources into a single cleaned dataset.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict


def consolidate_jobs(data_dir: Path = None, output_format: str = "parquet") -> Dict:
    """
    Consolidate all raw job data into interim and processed datasets.

    Process:
    1. Load all raw JSON files from data/raw/adzuna/
    2. Deduplicate by job ID
    3. Clean and standardize fields
    4. Save to data/interim/adzuna.parquet
    5. Consolidate all sources to data/processed/jobs.parquet

    Args:
        data_dir: Root data directory (auto-detected if None)
        output_format: Output format ('parquet' or 'csv')

    Returns:
        Dict with consolidation statistics
    """
    if data_dir is None:
        # Auto-detect project root
        project_root = Path(__file__).parent.parent.parent.parent
        data_dir = project_root / "data"

    raw_dir = data_dir / "raw"
    interim_dir = data_dir / "interim"
    processed_dir = data_dir / "processed"

    # Ensure output directories exist
    interim_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    consolidation_start = datetime.now()

    print(f"\n{'='*60}")
    print(f"Starting Data Consolidation")
    print(f"{'='*60}\n")

    # Step 1: Consolidate Adzuna data
    print("1. Processing Adzuna data...")
    adzuna_df = _consolidate_source(raw_dir / "adzuna", "adzuna")

    if adzuna_df is not None and len(adzuna_df) > 0:
        # Save to interim
        interim_path = interim_dir / f"adzuna.{output_format}"
        if output_format == "parquet":
            adzuna_df.to_parquet(interim_path, index=False)
        else:
            adzuna_df.to_csv(interim_path, index=False)

        print(f"   ✓ Saved {len(adzuna_df)} jobs to {interim_path}")
    else:
        print(f"   ✗ No Adzuna data found")
        adzuna_df = pd.DataFrame()

    # Step 2: Consolidate RemoteOK data
    print("\n2. Processing RemoteOK data...")
    remoteok_df = _consolidate_source(raw_dir / "remoteok", "remoteok")

    if remoteok_df is not None and len(remoteok_df) > 0:
        # Save to interim
        interim_path = interim_dir / f"remoteok.{output_format}"
        if output_format == "parquet":
            remoteok_df.to_parquet(interim_path, index=False)
        else:
            remoteok_df.to_csv(interim_path, index=False)

        print(f"   ✓ Saved {len(remoteok_df)} jobs to {interim_path}")
    else:
        print(f"   ✗ No RemoteOK data found")
        remoteok_df = pd.DataFrame()

    # Step 3: Consolidate all sources into final dataset
    print("\n3. Creating final consolidated dataset...")

    # Combine all sources
    dfs_to_merge = []
    if len(adzuna_df) > 0:
        dfs_to_merge.append(adzuna_df)
    if len(remoteok_df) > 0:
        dfs_to_merge.append(remoteok_df)

    if dfs_to_merge:
        final_df = pd.concat(dfs_to_merge, ignore_index=True)
    else:
        final_df = pd.DataFrame()

    if len(final_df) > 0:
        # Apply final cleaning
        final_df = _apply_final_cleaning(final_df)

        # Save to processed
        processed_path = processed_dir / f"jobs.{output_format}"
        if output_format == "parquet":
            final_df.to_parquet(processed_path, index=False)
        else:
            final_df.to_csv(processed_path, index=False)

        print(f"   ✓ Saved {len(final_df)} jobs to {processed_path}")
        _print_diversity_metrics(final_df)
    else:
        print(f"   ✗ No data to consolidate")

    consolidation_end = datetime.now()
    duration = (consolidation_end - consolidation_start).total_seconds()

    # Summary
    total_records_in = (
        (len(adzuna_df) if adzuna_df is not None and len(adzuna_df) > 0 else 0) +
        (len(remoteok_df) if remoteok_df is not None and len(remoteok_df) > 0 else 0)
    )

    print(f"\n{'='*60}")
    print(f"Consolidation Complete")
    print(f"{'='*60}")
    print(f"Duration: {duration:.1f}s")
    print(f"Records In: {total_records_in} (Adzuna: {len(adzuna_df) if adzuna_df is not None else 0}, RemoteOK: {len(remoteok_df) if remoteok_df is not None else 0})")
    print(f"Records Out: {len(final_df)}")
    print(f"Output: {processed_path if len(final_df) > 0 else 'None'}")
    print(f"{'='*60}\n")

    return {
        'timestamp': consolidation_start.isoformat(),
        'duration_seconds': duration,
        'status': 'success' if len(final_df) > 0 else 'no_data',
        'records_in': total_records_in,
        'records_in_by_source': {
            'adzuna': len(adzuna_df) if adzuna_df is not None else 0,
            'remoteok': len(remoteok_df) if remoteok_df is not None else 0
        },
        'records_out': len(final_df),
        'output_path': str(processed_path) if len(final_df) > 0 else None
    }


def _consolidate_source(source_raw_dir: Path, source_name: str) -> pd.DataFrame:
    """Load and consolidate all JSON files from a source directory."""
    if not source_raw_dir.exists():
        return pd.DataFrame()

    all_jobs = []

    # Recursively find all JSON files
    json_files = list(source_raw_dir.rglob("jobs_*.json"))

    if not json_files:
        return pd.DataFrame()

    print(f"   Found {len(json_files)} {source_name} files")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_jobs.extend(data)
                elif isinstance(data, dict):
                    all_jobs.append(data)
        except Exception as e:
            print(f"   Warning: Error loading {json_file}: {e}")

    if not all_jobs:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_jobs)

    # Deduplicate by job ID
    initial_count = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    duplicates_removed = initial_count - len(df)

    print(f"   Loaded {initial_count} records, removed {duplicates_removed} duplicates")

    return df


def _apply_final_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply final cleaning and standardization to consolidated data."""

    # Sort by created_date (newest first)
    if 'created_date' in df.columns:
        df = df.sort_values('created_date', ascending=False)

    # Remove jobs with missing critical fields
    initial_count = len(df)
    df = df.dropna(subset=['title', 'description'])
    removed = initial_count - len(df)

    if removed > 0:
        print(f"   Removed {removed} jobs with missing title/description")

    # Standardize text fields (trim whitespace, etc.)
    text_columns = ['title', 'company', 'description']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.strip()

    return df


def _print_diversity_metrics(df: pd.DataFrame) -> None:
    """
    Print quick diversity metrics for the consolidated dataset.

    Mirrors the manual checks in docs/DIVERSE_COLLECTION_STRATEGY.md so
    the processing pipeline surfaces coverage without any extra steps.
    """
    print(f"\n{'='*60}")
    print("Dataset Diversity Snapshot")
    print(f"{'='*60}")

    def _print_value_counts(column: str, label: str, top_n: int = 5):
        if column in df.columns:
            counts = df[column].fillna("Unknown").value_counts().head(top_n)
            print(f"\n{label}:")
            for value, count in counts.items():
                print(f"  - {value}: {count}")

    _print_value_counts('source', 'By Source')
    _print_value_counts('location_country', 'By Country')
    _print_value_counts('category_label', 'By Category')

    print(f"\nTotal jobs counted: {len(df)}")
