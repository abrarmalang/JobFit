"""
Main Collection Script

Orchestrates data collection from Adzuna API.
Run with: python scripts/run_collection_adzuna.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import shared
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.config import load_config
from .collector_adzuna import AdzunaCollector
from ..metrics import (
    log_collection_run,
    log_source_health,
    update_worker_status
)


def main():
    """Run Adzuna data collection."""

    # Load configuration from .env
    config = load_config()
    adzuna_config = config.adzuna

    # Validate API credentials
    if not adzuna_config.app_id or not adzuna_config.app_key:
        print("Error: ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in .env file")
        print("\nCreate a .env file in project root with:")
        print("ADZUNA_APP_ID=your_app_id")
        print("ADZUNA_APP_KEY=your_app_key")
        print("\nGet credentials at: https://developer.adzuna.com/")
        print("\nOptionally configure collection settings:")
        print("ADZUNA_COUNTRY=gb")
        print("ADZUNA_WHAT=software engineer python")
        print("ADZUNA_WHERE=London")
        print("ADZUNA_CATEGORY=it-jobs")
        print("ADZUNA_MAX_DAYS_OLD=7")
        print("ADZUNA_MAX_JOBS=100")
        sys.exit(1)

    update_worker_status(
        "collection",
        "RUNNING",
        {
            "mode": "single-profile",
            "started_at": datetime.now().isoformat()
        }
    )

    # Initialize collector
    collector = AdzunaCollector(
        app_id=adzuna_config.app_id,
        app_key=adzuna_config.app_key
    )

    # Run collection with config settings
    result = collector.fetch_jobs(
        country=adzuna_config.country,
        what=adzuna_config.what,
        where=adzuna_config.where,
        category=adzuna_config.category,
        max_days_old=adzuna_config.max_days_old,
        max_jobs=adzuna_config.max_jobs,
        results_per_page=adzuna_config.results_per_page
    )

    log_collection_run("Adzuna", "Config Search", result)

    response_time_ms = None
    if result['api_calls'] > 0:
        response_time_ms = (result['duration_seconds'] / result['api_calls']) * 1000

    log_source_health(
        "Adzuna",
        "ONLINE" if result['status'] in ("success", "no_new_jobs") else "ERROR",
        response_time_ms=response_time_ms,
        rate_limit_used=result.get('api_calls'),
        rate_limit_total=1000,
        details={
            "last_parameters": result.get('parameters', {})
        }
    )

    update_worker_status(
        "collection",
        "IDLE",
        {
            "last_run": result['timestamp'],
            "jobs_collected": result['jobs_collected'],
            "jobs_deduplicated": result['jobs_deduplicated']
        }
    )

    # Show statistics
    stats = collector.get_statistics()
    print(f"\nOverall Statistics:")
    print(f"Total Unique Jobs: {stats['unique_jobs']}")
    print(f"Collection Files: {stats['total_collection_files']}")
    print(f"Data Directory: {stats['raw_data_directory']}")

    return result


if __name__ == "__main__":
    main()
