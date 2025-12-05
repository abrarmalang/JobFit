"""
Parquet file utilities for handling large files.

Provides functions to split and load parquet files to avoid GitHub file size limits.
"""

from pathlib import Path
from typing import Union
import pandas as pd


MAX_FILE_SIZE_MB = 80  # Maximum file size in MB for GitHub


def save_parquet_chunked(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    max_size_mb: int = MAX_FILE_SIZE_MB,
    **kwargs
) -> None:
    """
    Save a DataFrame to parquet, splitting into chunks if needed.

    If the file would exceed max_size_mb, splits into multiple files:
    - base_name_part_0.parquet
    - base_name_part_1.parquet
    - etc.

    Args:
        df: DataFrame to save
        output_path: Path where to save the file(s)
        max_size_mb: Maximum file size in MB before splitting
        **kwargs: Additional arguments passed to to_parquet()
    """
    output_path = Path(output_path)

    # First, try saving as a single file to check size
    temp_path = output_path.parent / f".temp_{output_path.name}"
    df.to_parquet(temp_path, **kwargs)

    file_size_mb = temp_path.stat().st_size / (1024 * 1024)

    if file_size_mb <= max_size_mb:
        # File is small enough, just rename it
        temp_path.rename(output_path)
        print(f"✓ Saved {output_path.name} ({file_size_mb:.1f} MB)")
    else:
        # File is too large, need to split
        temp_path.unlink()  # Remove temp file

        # Calculate number of chunks needed
        num_chunks = int((file_size_mb / max_size_mb) + 1)
        chunk_size = len(df) // num_chunks + 1

        # Remove .parquet extension to add part number
        base_name = output_path.stem

        print(f"⚠ File would be {file_size_mb:.1f} MB, splitting into {num_chunks} parts...")

        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(df))
            chunk = df.iloc[start_idx:end_idx]

            chunk_path = output_path.parent / f"{base_name}_part_{i}.parquet"
            chunk.to_parquet(chunk_path, **kwargs)

            chunk_size_mb = chunk_path.stat().st_size / (1024 * 1024)
            print(f"  ✓ Saved {chunk_path.name} ({chunk_size_mb:.1f} MB, {len(chunk):,} rows)")


def load_parquet_chunked(base_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load a parquet file that may have been split into chunks.

    Tries to load:
    1. base_path directly if it exists
    2. base_path_part_0.parquet, base_path_part_1.parquet, etc. if split

    Args:
        base_path: Base path to the parquet file

    Returns:
        Combined DataFrame
    """
    base_path = Path(base_path)

    # Check if single file exists
    if base_path.exists():
        return pd.read_parquet(base_path)

    # Check for chunked files
    base_name = base_path.stem
    parent_dir = base_path.parent

    chunk_files = sorted(parent_dir.glob(f"{base_name}_part_*.parquet"))

    if not chunk_files:
        raise FileNotFoundError(
            f"Could not find {base_path} or chunked files {base_name}_part_*.parquet"
        )

    print(f"Loading {len(chunk_files)} parquet chunks...")
    chunks = []
    for chunk_file in chunk_files:
        chunk = pd.read_parquet(chunk_file)
        chunks.append(chunk)
        print(f"  ✓ Loaded {chunk_file.name} ({len(chunk):,} rows)")

    df = pd.concat(chunks, ignore_index=True)
    print(f"✓ Combined into single DataFrame with {len(df):,} rows")

    return df
