#!/usr/bin/env python
"""
Run Diverse Data Collection

Collects jobs from multiple countries, categories, and roles to create
a diverse dataset for testing matching accuracy.

Usage:
    python scripts/run_collection_diverse.py
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


# Collection profiles for diversity
COLLECTION_PROFILES = [
    # UK Tech Jobs (broad search for organic mix)
    {
        "name": "UK - Mixed Tech Roles",
        "country": "gb",
        "what": "",
        "category": "it-jobs",
        "max_jobs": 60
    },
    {
        "name": "UK - Data Science",
        "country": "gb",
        "what": "data scientist",
        "category": "it-jobs",
        "max_jobs": 50
    },
    {
        "name": "UK - Frontend Development",
        "country": "gb",
        "what": "frontend developer react",
        "category": "it-jobs",
        "max_jobs": 30
    },
    {
        "name": "UK - Backend Development",
        "country": "gb",
        "what": "backend developer python",
        "category": "it-jobs",
        "max_jobs": 30
    },

    # US Tech Jobs
    {
        "name": "US - Software Engineering",
        "country": "us",
        "what": "software engineer",
        "category": "it-jobs",
        "max_jobs": 50
    },
    {
        "name": "US - Machine Learning",
        "country": "us",
        "what": "machine learning engineer",
        "category": "it-jobs",
        "max_jobs": 40
    },

    # Australia Tech Jobs
    {
        "name": "AU - Software Engineering",
        "country": "au",
        "what": "software engineer",
        "category": "it-jobs",
        "max_jobs": 30
    },

    # Different Categories (to test cross-domain matching)
    {
        "name": "UK - Product Management",
        "country": "gb",
        "what": "product manager",
        "category": "it-jobs",
        "max_jobs": 20
    },
    {
        "name": "UK - DevOps",
        "country": "gb",
        "what": "devops engineer",
        "category": "it-jobs",
        "max_jobs": 30
    },
    {
        "name": "UK - Mobile Development",
        "country": "gb",
        "what": "mobile developer",
        "category": "it-jobs",
        "max_jobs": 20
    },
]


def main():
    """Run diverse data collection across multiple profiles."""

    # Load config for API credentials
    config = load_config()
    adzuna_config = config.adzuna

    # Validate API credentials
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
            "mode": "diverse-profiles",
            "profiles": len(COLLECTION_PROFILES),
            "started_at": datetime.now().isoformat()
        }
    )

    # Initialize collector
    collector = AdzunaCollector(
        app_id=adzuna_config.app_id,
        app_key=adzuna_config.app_key
    )

    print(f"\n{'='*70}")
    print(f"DIVERSE DATA COLLECTION")
    print(f"{'='*70}")
    print(f"Collecting from {len(COLLECTION_PROFILES)} different profiles")
    print(f"This will create a diverse dataset for testing matching accuracy\n")

    total_jobs_collected = 0
    results = []

    # Run collection for each profile
    last_timestamp = None

    for i, profile in enumerate(COLLECTION_PROFILES, 1):
        print(f"\n[{i}/{len(COLLECTION_PROFILES)}] {profile['name']}")
        print(f"{'─'*70}")

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

        total_jobs_collected += result['jobs_collected']
        results.append({
            'profile': profile['name'],
            'jobs_collected': result['jobs_collected'],
            'jobs_deduplicated': result['jobs_deduplicated']
        })
        last_timestamp = result.get('timestamp')

    # Summary
    print(f"\n{'='*70}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Profile':<40} {'New Jobs':<15} {'Duplicates':<15}")
    print(f"{'-'*70}")
    for result in results:
        print(f"{result['profile']:<40} {result['jobs_collected']:<15} {result['jobs_deduplicated']:<15}")

    print(f"{'-'*70}")
    print(f"{'TOTAL':<40} {total_jobs_collected:<15}")

    # Overall statistics
    stats = collector.get_statistics()
    print(f"\n{'='*70}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*70}")
    print(f"Total Unique Jobs: {stats['unique_jobs']}")
    print(f"Collection Files: {stats['total_collection_files']}")
    print(f"Data Directory: {stats['raw_data_directory']}")
    print(f"\nNext step: Run 'python scripts/run_processing.py' to consolidate data")
    print(f"{'='*70}\n")

    update_worker_status(
        "collection",
        "IDLE",
        {
            "mode": "diverse-profiles",
            "profiles": len(COLLECTION_PROFILES),
            "last_run": last_timestamp,
            "jobs_collected": total_jobs_collected
        }
    )

    return results


if __name__ == "__main__":
    main()
