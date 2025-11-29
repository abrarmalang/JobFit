"""
Adzuna API Collector

Fetches job postings from Adzuna API with deduplication and date filtering.
"""

import requests
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set

from .collector_base import CollectorBase


class AdzunaCollector(CollectorBase):
    """Collector for fetching jobs from Adzuna API."""

    def __init__(self, app_id: str, app_key: str, data_dir: Path = None):
        """
        Initialize Adzuna collector.

        Args:
            app_id: Adzuna API application ID
            app_key: Adzuna API application key
            data_dir: Root data directory (auto-detected if None)
        """
        super().__init__(data_dir)

        self.app_id = app_id
        self.app_key = app_key
        self.base_url = "https://api.adzuna.com/v1/api/jobs"

        # Adzuna-specific raw directory
        self.adzuna_raw_dir = self.raw_dir / "adzuna"
        self.adzuna_raw_dir.mkdir(parents=True, exist_ok=True)

        # Load existing job IDs for deduplication
        self.existing_job_ids = self._load_existing_job_ids()

    def _load_existing_job_ids(self) -> Set[str]:
        """Load all existing job IDs from stored data for deduplication."""
        job_ids = set()

        # Scan all JSON files in adzuna raw directory
        for json_file in self.adzuna_raw_dir.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        job_ids.update([str(job.get('id')) for job in data if job.get('id')])
                    elif isinstance(data, dict) and 'id' in data:
                        job_ids.add(str(data['id']))
            except Exception as e:
                print(f"Warning: Error loading {json_file}: {e}")

        if job_ids:
            print(f"✓ Loaded {len(job_ids)} existing job IDs for deduplication")

        return job_ids

    def fetch_jobs(
        self,
        country: str = "gb",
        what: str = "",
        where: str = "",
        category: Optional[str] = None,
        max_days_old: int = 7,
        max_jobs: int = 100,
        results_per_page: int = 50
    ) -> Dict:
        """
        Fetch jobs from Adzuna API with date filtering and deduplication.

        Args:
            country: Country code (gb, us, au, etc.)
            what: Job search keywords (e.g., "software engineer", "data scientist")
            where: Location filter (e.g., "London", "Remote")
            category: Job category filter (e.g., 'it-jobs')
            max_days_old: Only fetch jobs posted within last N days
            max_jobs: Maximum number of new jobs to fetch
            results_per_page: Results per API call (max 50)

        Returns:
            Dict with collection statistics and metadata
        """
        collection_start = datetime.now()
        jobs_collected = []
        jobs_deduplicated = 0
        api_calls = 0
        page = 1

        print(f"\n{'='*60}")
        print(f"Starting Adzuna Collection")
        print(f"{'='*60}")
        print(f"Country: {country}")
        print(f"Search: {what or 'all jobs'}")
        print(f"Location: {where or 'all locations'}")
        print(f"Category: {category or 'all categories'}")
        print(f"Max days old: {max_days_old}")
        print(f"Target: {max_jobs} new jobs")
        print(f"{'='*60}\n")

        while len(jobs_collected) < max_jobs:
            url = f"{self.base_url}/{country}/search/{page}"

            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "results_per_page": min(results_per_page, 50),
                "sort_by": "date",  # Get newest first
                "max_days_old": max_days_old
            }

            if what:
                params["what"] = what
            if where:
                params["where"] = where
            if category:
                params["category"] = category

            try:
                api_calls += 1
                print(f"Fetching page {page}...", end=" ")
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                results = data.get('results', [])
                if not results:
                    print(f"✗ No more results")
                    break

                print(f"✓ Got {len(results)} results")

                # Process jobs and deduplicate
                new_jobs_this_page = 0
                for job in results:
                    job_id = str(job.get('id'))

                    # Skip if already exists
                    if job_id in self.existing_job_ids:
                        jobs_deduplicated += 1
                        continue

                    # Structure and add job
                    structured_job = self._structure_job_data(job)
                    jobs_collected.append(structured_job)
                    self.existing_job_ids.add(job_id)
                    new_jobs_this_page += 1

                    if len(jobs_collected) >= max_jobs:
                        break

                print(f"  → {new_jobs_this_page} new, {len(results) - new_jobs_this_page} duplicates")

                # Check if we've reached the end
                if len(results) < results_per_page:
                    print(f"✓ Reached end of available results")
                    break

                page += 1

            except requests.exceptions.RequestException as e:
                print(f"\n✗ API error on page {page}: {e}")
                break
            except Exception as e:
                print(f"\n✗ Unexpected error on page {page}: {e}")
                break

        collection_end = datetime.now()
        duration = (collection_end - collection_start).total_seconds()

        # Save collected jobs to raw directory
        filepath = None
        if jobs_collected:
            filepath = self._save_raw_data(jobs_collected, "adzuna", collection_start)

        # Collection summary
        print(f"\n{'='*60}")
        print(f"Collection Complete")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f}s")
        print(f"API Calls: {api_calls}")
        print(f"New Jobs: {len(jobs_collected)}")
        print(f"Duplicates Skipped: {jobs_deduplicated}")
        print(f"Total Unique Jobs: {len(self.existing_job_ids)}")
        if filepath:
            print(f"Saved to: {filepath}")
        print(f"{'='*60}\n")

        # Return collection metadata
        return {
            'timestamp': collection_start.isoformat(),
            'duration_seconds': duration,
            'status': 'success' if jobs_collected else 'no_new_jobs',
            'jobs_collected': len(jobs_collected),
            'jobs_deduplicated': jobs_deduplicated,
            'api_calls': api_calls,
            'parameters': {
                'country': country,
                'what': what,
                'where': where,
                'category': category,
                'max_days_old': max_days_old,
                'max_jobs': max_jobs
            },
            'filepath': str(filepath) if filepath else None,
            'job_ids': [j['id'] for j in jobs_collected]
        }

    def _structure_job_data(self, raw_job: Dict) -> Dict:
        """
        Structure raw Adzuna job data into standardized format.

        Args:
            raw_job: Raw job data from Adzuna API

        Returns:
            Structured job dictionary
        """
        # Calculate job age
        created_date = raw_job.get('created')
        job_age_days = None
        if created_date:
            try:
                created = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                job_age_days = (datetime.now(created.tzinfo) - created).days
            except:
                pass

        # Parse location hierarchy
        location = raw_job.get('location', {})
        location_area = location.get('area', [])

        return {
            # Core fields
            'id': str(raw_job.get('id')),
            'title': raw_job.get('title'),
            'company': raw_job.get('company', {}).get('display_name', 'Unknown'),
            'description': raw_job.get('description', ''),
            'redirect_url': raw_job.get('redirect_url'),

            # Category
            'category_tag': raw_job.get('category', {}).get('tag'),
            'category_label': raw_job.get('category', {}).get('label'),

            # Location
            'location_display': location.get('display_name'),
            'location_country': location_area[0] if len(location_area) > 0 else None,
            'location_region': location_area[1] if len(location_area) > 1 else None,
            'location_city': location_area[2] if len(location_area) > 2 else None,
            'latitude': raw_job.get('latitude'),
            'longitude': raw_job.get('longitude'),

            # Salary
            'salary_min': raw_job.get('salary_min'),
            'salary_max': raw_job.get('salary_max'),
            'salary_is_predicted': raw_job.get('salary_is_predicted', '1') == '0',

            # Employment
            'contract_type': raw_job.get('contract_type'),
            'contract_time': raw_job.get('contract_time'),
            'created_date': created_date,
            'job_age_days': job_age_days,

            # Metadata
            'source': 'adzuna',
            'collected_at': datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict:
        """Get overall collection statistics."""
        total_jobs_files = list(self.adzuna_raw_dir.rglob("jobs_*.json"))

        return {
            'unique_jobs': len(self.existing_job_ids),
            'total_collection_files': len(total_jobs_files),
            'raw_data_directory': str(self.adzuna_raw_dir)
        }
