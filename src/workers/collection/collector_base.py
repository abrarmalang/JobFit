"""
Base Collector Class

Abstract base class for all data collectors to ensure consistent interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, List
from pathlib import Path
from datetime import datetime


class CollectorBase(ABC):
    """Abstract base class for job data collectors."""

    def __init__(self, data_dir: Path = None):
        """
        Initialize collector with data directory.

        Args:
            data_dir: Root data directory (defaults to project data/ folder)
        """
        if data_dir is None:
            # Auto-detect project root and use data/ folder
            # Use resolve() for absolute path and parents[3] for clarity
            self.data_dir = Path(__file__).resolve().parents[3] / "data"
        else:
            self.data_dir = data_dir

        self.raw_dir = self.data_dir / "raw"
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def fetch_jobs(self, **kwargs) -> Dict:
        """
        Fetch jobs from the data source.

        Returns:
            Dict containing collection metadata and statistics
        """
        pass

    @abstractmethod
    def _structure_job_data(self, raw_job: Dict) -> Dict:
        """
        Structure raw job data into standardized format.

        Args:
            raw_job: Raw job data from source

        Returns:
            Structured job dictionary
        """
        pass

    def save_raw_list(self, jobs: List[Dict], source: str, silent: bool = False):
        """
        Save a full list of jobs into a JSON file.

        Args:
            jobs: List of job dictionaries to save
            source: Name of the data source (e.g., 'adzuna', 'scraped')
            silent: If True, don't print confirmation message

        Returns:
            Path to the saved file
        """
        import json

        # Create source-specific directory with date
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        source_dir = self.raw_dir / source / date_str
        source_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        filename = f"jobs_{now.strftime('%H%M%S')}.json"
        filepath = source_dir / filename

        # Save as JSON (optimized: no indent for smaller files)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(jobs, f, ensure_ascii=False)

        if not silent:
            print(f"Saved {len(jobs)} jobs to {filepath}")

        return filepath

    def save_raw_job(self, job: Dict):
        """
        Append a single structured job to a daily JSONL file.

        Args:
            job: Single job dictionary to save
        """
        import json

        # Create source-specific directory with date
        now = datetime.now()
        date_str = now.strftime('%Y-%m-%d')
        source = job.get('source', 'unknown')
        source_dir = self.raw_dir / source / date_str
        source_dir.mkdir(parents=True, exist_ok=True)

        # Use daily JSONL file (one line per job)
        filename = f"jobs_{date_str}.jsonl"
        filepath = source_dir / filename

        # Append to JSONL
        with open(filepath, 'a', encoding='utf-8') as f:
            json.dump(job, f, ensure_ascii=False)
            f.write('\n')

    def _save_raw_data(self, data: List[Dict], source_name: str, collection_time: datetime):
        """
        Legacy method - redirects to save_raw_list for compatibility.

        Args:
            data: List of job dictionaries to save
            source_name: Name of the data source (e.g., 'adzuna', 'scraped')
            collection_time: Timestamp of collection (not used, uses current time)
        """
        return self.save_raw_list(data, source_name)
