"""
Main Processing Script

Orchestrates data cleaning and consolidation.
Run with: python -m src.jobfit_workers.processing.main_processing
"""

from datetime import datetime

from .consolidation_data import consolidate_jobs
from ..metrics import log_processing_run, update_worker_status


def main():
    """Run data consolidation pipeline."""

    print("Starting data processing pipeline...")

    update_worker_status(
        "processing",
        "RUNNING",
        {
            "started_at": datetime.now().isoformat()
        }
    )

    # Run consolidation
    result = consolidate_jobs(output_format="parquet")

    log_processing_run(result)

    update_worker_status(
        "processing",
        "IDLE",
        {
            "last_run": result['timestamp'],
            "records_out": result['records_out']
        }
    )

    if result['status'] == 'success':
        print(f"\n✓ Processing complete!")
        print(f"Output file: {result['output_path']}")
    else:
        print(f"\n✗ No data to process")

    return result


if __name__ == "__main__":
    main()
