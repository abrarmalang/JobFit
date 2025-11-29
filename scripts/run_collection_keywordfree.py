#!/usr/bin/env python
"""
Run Keyword-Free UK Collection

Fetches a broad mix of UK tech roles without specifying keywords so the
dataset includes organic variation beyond the targeted role profiles.

Usage:
    python scripts/run_collection_keywordfree.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from workers.collection.collector_adzuna import AdzunaCollector
from shared.config import load_config
from workers.metrics import (
    log_collection_run,
    log_source_health,
    update_worker_status
)


def main():
    """Run a keyword-free UK Adzuna collection."""

    config = load_config()
    adzuna_config = config.adzuna

    if not adzuna_config.app_id or not adzuna_config.app_key:
        print("Error: ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in .env file")
        print("\nCreate a .env file in project root with:")
        print("ADZUNA_APP_ID=your_app_id")
        print("ADZUNA_APP_KEY=your_app_key")
        sys.exit(1)

    update_worker_status(
        "collection",
        "RUNNING",
        {
            "mode": "keyword-free-uk",
            "started_at": datetime.now().isoformat()
        }
    )

    collector = AdzunaCollector(
        app_id=adzuna_config.app_id,
        app_key=adzuna_config.app_key
    )

    profile = {
        "name": "UK - Mixed Tech Roles (keyword-free)",
        "country": "gb",
        "what": "",
        "category": "it-jobs",
        "max_jobs": 60
    }

    print(f"\n{'='*70}")
    print(profile["name"])
    print(f"{'='*70}")
    print("Collecting a natural mix of roles without keyword filters\n")

    result = collector.fetch_jobs(
        country=profile["country"],
        what=profile["what"],
        where=profile.get("where", ""),
        category=profile.get("category", "it-jobs"),
        max_days_old=adzuna_config.max_days_old,
        max_jobs=profile["max_jobs"],
        results_per_page=adzuna_config.results_per_page
    )

    log_collection_run("Adzuna", profile["name"], result)

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
            "profile": profile["name"]
        }
    )

    print(f"\nNew Jobs Collected: {result['jobs_collected']}")
    print(f"Duplicates Skipped: {result['jobs_deduplicated']}")

    stats = collector.get_statistics()
    print(f"\n{'='*70}")
    print("OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total Unique Jobs: {stats['unique_jobs']}")
    print(f"Collection Files: {stats['total_collection_files']}")
    print(f"Data Directory: {stats['raw_data_directory']}")
    print(f"{'='*70}\n")

    update_worker_status(
        "collection",
        "IDLE",
        {
            "mode": "keyword-free-uk",
            "last_run": result['timestamp'],
            "jobs_collected": result['jobs_collected']
        }
    )

    return result


if __name__ == "__main__":
    main()
