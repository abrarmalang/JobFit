#!/usr/bin/env python
"""
Run RemoteOK Data Collection

Cross-platform script to fetch jobs from RemoteOK API.

Usage:
    python scripts/run_collection_remoteok.py
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from workers.collection.collector_remoteok import RemoteOKCollector
from workers.metrics import (
    log_collection_run,
    log_source_health,
    update_worker_status
)


def main():
    """Run RemoteOK data collection."""

    print("Starting RemoteOK data collection...")
    print("Note: RemoteOK API is free and requires no authentication")
    print("Please link back to RemoteOK.com as per API terms\n")

    update_worker_status(
        "collection",
        "RUNNING",
        {
            "mode": "remoteok",
            "started_at": datetime.now().isoformat()
        }
    )

    # Initialize collector (no API key needed!)
    collector = RemoteOKCollector()

    # Run collection with optional tag filtering
    # Examples of useful tags: 'python', 'react', 'javascript', 'devops', 'data', 'engineer'
    result = collector.fetch_jobs(
        max_jobs=100,  # Collect up to 100 new jobs
        filter_tags=['python', 'react', 'javascript', 'engineer']  # Optional: filter by tags
        # filter_tags=None  # Or set to None to get all jobs
    )

    log_collection_run("RemoteOK", "Remote Roles", result)

    response_time_ms = result['duration_seconds'] * 1000
    log_source_health(
        "RemoteOK",
        "ONLINE" if result['status'] in ("success", "no_new_jobs") else "ERROR",
        response_time_ms=response_time_ms,
        rate_limit_used=None,
        rate_limit_total=None,
        details={
            "filter_tags": result['parameters'].get('filter_tags')
        }
    )

    update_worker_status(
        "collection",
        "IDLE",
        {
            "mode": "remoteok",
            "last_run": result['timestamp'],
            "jobs_collected": result['jobs_collected']
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
