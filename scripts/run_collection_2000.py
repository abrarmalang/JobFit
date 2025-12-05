#!/usr/bin/env python
"""
Run 2000 Job Collection

Collects 2000 jobs from Adzuna API across multiple countries.
Includes rate limiting, delays, and automatic retry on 429 errors.

Usage:
    python scripts/run_collection_2000.py
"""

import sys
import time
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


# Collection profiles for 2000 jobs total
COLLECTION_PROFILES_2000 = [
    # UK - Main volume
    {
        "name": "UK - All Jobs",
        "country": "gb",
        "what": "",
        "category": None,
        "max_jobs": 800,
        "max_days_old": 60,
    },

    # US - Large market
    {
        "name": "US - All Jobs",
        "country": "us",
        "what": "",
        "category": None,
        "max_jobs": 600,
        "max_days_old": 60,
    },

    # Australia
    {
        "name": "AU - All Jobs",
        "country": "au",
        "what": "",
        "category": None,
        "max_jobs": 400,
        "max_days_old": 60,
    },

    # Canada
    {
        "name": "CA - All Jobs",
        "country": "ca",
        "what": "",
        "category": None,
        "max_jobs": 200,
        "max_days_old": 60,
    },
]


def main():
    """Run 2000 job collection with rate limiting."""

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
            "mode": "2000-jobs",
            "profiles": len(COLLECTION_PROFILES_2000),
            "target_jobs": sum(p["max_jobs"] for p in COLLECTION_PROFILES_2000),
            "started_at": datetime.now().isoformat()
        }
    )

    # Initialize collector
    collector = AdzunaCollector(
        app_id=adzuna_config.app_id,
        app_key=adzuna_config.app_key,
        max_pages=100,          # Sufficient for 2000 jobs
        max_workers=1,          # Single worker to avoid rate limits
        batch_save_size=500     # Save every 500 jobs
    )

    print(f"\n{'='*80}")
    print(f"2000 JOB COLLECTION")
    print(f"{'='*80}")
    print(f"Target: ~{sum(p['max_jobs'] for p in COLLECTION_PROFILES_2000):,} jobs")
    print(f"Profiles: {len(COLLECTION_PROFILES_2000)}")
    print(f"Max pages per profile: 100")
    print(f"Concurrent workers: 1 (sequential to avoid rate limits)")
    print(f"Batch save size: 500 jobs")
    print(f"Results per page: {adzuna_config.results_per_page}")
    print(f"\nThis will take approximately 10-15 minutes depending on API response times.")
    print(f"Progress will be shown after each profile.\n")

    total_jobs_collected = 0
    total_api_calls = 0
    results = []
    start_time = datetime.now()

    # Run collection for each profile
    last_timestamp = None

    for i, profile in enumerate(COLLECTION_PROFILES_2000, 1):
        print(f"\n{'='*80}")
        print(f"[{i}/{len(COLLECTION_PROFILES_2000)}] {profile['name']}")
        print(f"{'='*80}")
        print(f"Target: {profile['max_jobs']:,} jobs | Max days: {profile['max_days_old']}")

        profile_start = datetime.now()

        try:
            result = collector.fetch_jobs(
                country=profile["country"],
                what=profile["what"],
                where=profile.get("where", ""),
                category=profile.get("category", "it-jobs"),
                max_days_old=profile.get("max_days_old", 60),
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
            total_api_calls += result.get('api_calls', 0)

            profile_duration = (datetime.now() - profile_start).total_seconds()

            results.append({
                'profile': profile['name'],
                'jobs_collected': result['jobs_collected'],
                'jobs_deduplicated': result['jobs_deduplicated'],
                'api_calls': result.get('api_calls', 0),
                'duration': profile_duration,
                'status': result['status']
            })

            last_timestamp = result.get('timestamp')

            print(f"\n✓ Profile complete:")
            print(f"  Jobs collected: {result['jobs_collected']:,}")
            print(f"  API calls: {result.get('api_calls', 0)}")
            print(f"  Duration: {profile_duration:.1f}s")
            print(f"  Running total: {total_jobs_collected:,} jobs")

            # Add delay between profiles to avoid rate limiting
            # Skip delay after last profile
            if i < len(COLLECTION_PROFILES_2000):
                delay_seconds = 5
                print(f"\n⏸  Waiting {delay_seconds}s before next profile (rate limit protection)...")
                time.sleep(delay_seconds)

        except Exception as e:
            print(f"\n❌ Error collecting profile '{profile['name']}': {e}")
            results.append({
                'profile': profile['name'],
                'jobs_collected': 0,
                'jobs_deduplicated': 0,
                'api_calls': 0,
                'duration': 0,
                'status': 'error'
            })
            continue

    total_duration = (datetime.now() - start_time).total_seconds()

    # Summary
    print(f"\n{'='*80}")
    print(f"COLLECTION SUMMARY")
    print(f"{'='*80}\n")

    print(f"{'Profile':<45} {'Jobs':<10} {'API Calls':<12} {'Time':<10} {'Status':<10}")
    print(f"{'-'*80}")
    for result in results:
        status_icon = "✓" if result['status'] == 'success' else "✗"
        print(f"{result['profile']:<45} {result['jobs_collected']:<10,} "
              f"{result['api_calls']:<12} {result['duration']:<10.1f}s "
              f"{status_icon} {result['status']:<10}")

    print(f"{'-'*80}")
    print(f"{'TOTAL':<45} {total_jobs_collected:<10,} {total_api_calls:<12} {total_duration:<10.1f}s")

    # Overall statistics
    stats = collector.get_statistics()
    print(f"\n{'='*80}")
    print(f"OVERALL STATISTICS")
    print(f"{'='*80}")
    print(f"Total Unique Jobs: {total_jobs_collected:,}")
    print(f"Total API Calls: {total_api_calls}")
    print(f"Total Duration: {total_duration / 60:.1f} minutes")
    print(f"Avg Time per Profile: {total_duration / len(COLLECTION_PROFILES_2000):.1f}s")
    print(f"Data Directory: {stats['raw_data_directory']}")

    print(f"\n{'='*80}")
    print(f"NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Run processing:       python scripts/run_processing.py")
    print(f"2. Generate embeddings:  python src/workers/embeddings/generator.py")
    print(f"3. Train cluster model:  python src/workers/model_training/train_cluster_model.py")
    print(f"{'='*80}\n")

    update_worker_status(
        "collection",
        "IDLE",
        {
            "mode": "2000-jobs",
            "profiles": len(COLLECTION_PROFILES_2000),
            "last_run": last_timestamp,
            "jobs_collected": total_jobs_collected,
            "api_calls": total_api_calls,
            "duration_minutes": total_duration / 60
        }
    )

    return results


if __name__ == "__main__":
    main()
