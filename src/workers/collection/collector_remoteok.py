"""
RemoteOK API Collector

Fetches remote job postings from RemoteOK API with deduplication.
"""

import requests
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
import time

from .collector_base import CollectorBase


class RemoteOKCollector(CollectorBase):
    """Collector for fetching jobs from RemoteOK API."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize RemoteOK collector.

        Args:
            data_dir: Root data directory (auto-detected if None)
        """
        super().__init__(data_dir)

        self.api_url = "https://remoteok.com/api"

        # RemoteOK-specific raw directory
        self.remoteok_raw_dir = self.raw_dir / "remoteok"
        self.remoteok_raw_dir.mkdir(parents=True, exist_ok=True)

        # Load existing job IDs for deduplication
        self.existing_job_ids = self._load_existing_job_ids()

    def _load_existing_job_ids(self) -> Set[str]:
        """Load all existing job IDs from stored data for deduplication."""
        job_ids = set()

        # Scan all JSON files in remoteok raw directory
        for json_file in self.remoteok_raw_dir.rglob("*.json"):
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
        max_jobs: int = 100,
        filter_tags: List[str] = None
    ) -> Dict:
        """
        Fetch jobs from RemoteOK API with deduplication.

        RemoteOK returns all jobs in a single API call - no pagination needed.

        Args:
            max_jobs: Maximum number of new jobs to fetch
            filter_tags: Optional list of tags to filter by (e.g., ['python', 'react'])

        Returns:
            Dict with collection statistics and metadata
        """
        collection_start = datetime.now()
        jobs_collected = []
        jobs_deduplicated = 0
        jobs_filtered = 0

        print(f"\n{'='*60}")
        print(f"Starting RemoteOK Collection")
        print(f"{'='*60}")
        if filter_tags:
            print(f"Filter tags: {', '.join(filter_tags)}")
        print(f"Target: {max_jobs} new jobs")
        print(f"{'='*60}\n")

        try:
            # Fetch all jobs from RemoteOK API
            print(f"Fetching jobs from RemoteOK API...", end=" ")

            # Add User-Agent header (RemoteOK requires it)
            headers = {
                'User-Agent': 'Jobfit-Collector/1.0 (https://jobfit.app)'
            }

            response = requests.get(self.api_url, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            # First item is metadata/legal notice, skip it
            if isinstance(data, list) and len(data) > 0:
                # Check if first item is metadata (has 'legal' or 'last_updated' field)
                if isinstance(data[0], dict) and ('legal' in data[0] or 'last_updated' in data[0]):
                    metadata = data[0]
                    jobs_data = data[1:]
                    print(f"✓ Got {len(jobs_data)} jobs")

                    # Store legal notice for reference
                    if 'legal' in metadata:
                        legal_notice = metadata['legal']
                        print(f"\n📋 API Terms: Please link back to RemoteOK and mention as source")
                else:
                    jobs_data = data
                    print(f"✓ Got {len(jobs_data)} jobs")
            else:
                print(f"✗ Invalid response format")
                jobs_data = []

            # Process jobs
            for job in jobs_data:
                if not isinstance(job, dict):
                    continue

                job_id = str(job.get('id', ''))

                if not job_id:
                    continue

                # Skip if already exists
                if job_id in self.existing_job_ids:
                    jobs_deduplicated += 1
                    continue

                # Filter by tags if specified
                if filter_tags:
                    job_tags = job.get('tags', [])
                    # Check if any filter tag matches job tags
                    if not any(tag.lower() in [jt.lower() for jt in job_tags] for tag in filter_tags):
                        jobs_filtered += 1
                        continue

                # Structure and add job
                structured_job = self._structure_job_data(job)
                jobs_collected.append(structured_job)
                self.existing_job_ids.add(job_id)

                if len(jobs_collected) >= max_jobs:
                    break

            print(f"  → {len(jobs_collected)} new jobs collected")
            if jobs_deduplicated > 0:
                print(f"  → {jobs_deduplicated} duplicates skipped")
            if jobs_filtered > 0:
                print(f"  → {jobs_filtered} filtered by tags")

        except requests.exceptions.RequestException as e:
            print(f"\n✗ API error: {e}")
        except Exception as e:
            print(f"\n✗ Unexpected error: {e}")

        collection_end = datetime.now()
        duration = (collection_end - collection_start).total_seconds()

        # Save collected jobs to raw directory
        filepath = None
        if jobs_collected:
            filepath = self._save_raw_data(jobs_collected, "remoteok", collection_start)

        # Collection summary
        print(f"\n{'='*60}")
        print(f"Collection Complete")
        print(f"{'='*60}")
        print(f"Duration: {duration:.1f}s")
        print(f"New Jobs: {len(jobs_collected)}")
        print(f"Duplicates Skipped: {jobs_deduplicated}")
        if jobs_filtered > 0:
            print(f"Filtered Out: {jobs_filtered}")
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
            'jobs_filtered': jobs_filtered,
            'parameters': {
                'max_jobs': max_jobs,
                'filter_tags': filter_tags
            },
            'filepath': str(filepath) if filepath else None,
            'job_ids': [j['id'] for j in jobs_collected]
        }

    def _structure_job_data(self, raw_job: Dict) -> Dict:
        """
        Structure raw RemoteOK job data into standardized format.

        Args:
            raw_job: Raw job data from RemoteOK API

        Returns:
            Structured job dictionary matching our schema
        """
        # Parse date from epoch timestamp
        created_date = None
        job_age_days = None
        if raw_job.get('epoch'):
            try:
                created_dt = datetime.fromtimestamp(raw_job['epoch'])
                created_date = created_dt.isoformat()
                job_age_days = (datetime.now() - created_dt).days
            except:
                pass

        # Clean description (remove HTML tags for consistent format)
        description = raw_job.get('description', '')
        if description:
            # Basic HTML stripping (can use html.parser for more robust cleaning)
            import re
            # Remove HTML tags
            description = re.sub(r'<[^>]+>', ' ', description)
            # Remove extra whitespace
            description = re.sub(r'\s+', ' ', description).strip()

        # Extract salary info
        salary_min = raw_job.get('salary_min', 0)
        salary_max = raw_job.get('salary_max', 0)

        # Handle 0 values (RemoteOK uses 0 for unspecified)
        salary_min = salary_min if salary_min > 0 else None
        salary_max = salary_max if salary_max > 0 else None

        return {
            # Core fields
            'id': str(raw_job.get('id')),
            'title': raw_job.get('position', ''),
            'company': raw_job.get('company', 'Unknown'),
            'description': description,
            'redirect_url': raw_job.get('apply_url', raw_job.get('url', '')),

            # Category (using tags as categories)
            'category_tag': 'remote-work',  # All RemoteOK jobs are remote
            'category_label': 'Remote Work',
            'tags': raw_job.get('tags', []),  # Store original tags

            # Location
            'location_display': raw_job.get('location', 'Remote'),
            'location_country': None,  # RemoteOK doesn't always specify
            'location_region': None,
            'location_city': raw_job.get('location', 'Remote'),
            'latitude': None,  # Not provided by RemoteOK
            'longitude': None,

            # Salary
            'salary_min': salary_min,
            'salary_max': salary_max,
            'salary_is_predicted': False,  # RemoteOK doesn't predict salaries

            # Employment
            'contract_type': None,  # Not explicitly provided
            'contract_time': 'full-time',  # Most RemoteOK jobs are full-time
            'created_date': created_date,
            'job_age_days': job_age_days,

            # RemoteOK-specific fields
            'slug': raw_job.get('slug', ''),
            'company_logo': raw_job.get('company_logo', ''),
            'logo': raw_job.get('logo', ''),

            # Metadata
            'source': 'remoteok',
            'collected_at': datetime.now().isoformat()
        }

    def get_statistics(self) -> Dict:
        """Get overall collection statistics."""
        total_jobs_files = list(self.remoteok_raw_dir.rglob("jobs_*.json"))

        return {
            'unique_jobs': len(self.existing_job_ids),
            'total_collection_files': len(total_jobs_files),
            'raw_data_directory': str(self.remoteok_raw_dir)
        }
