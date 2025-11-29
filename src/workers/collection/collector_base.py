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
            project_root = Path(__file__).parent.parent.parent.parent
            data_dir = project_root / "data"

        self.data_dir = data_dir
        self.raw_dir = data_dir / "raw"
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

    def _save_raw_data(self, data: List[Dict], source_name: str, collection_time: datetime):
        """
        Save raw data to appropriate directory with timestamped filename.

        Args:
            data: List of job dictionaries to save
            source_name: Name of the data source (e.g., 'adzuna', 'scraped')
            collection_time: Timestamp of collection
        """
        import json

        # Create source-specific directory with date
        date_str = collection_time.strftime('%Y-%m-%d')
        source_dir = self.raw_dir / source_name / date_str
        source_dir.mkdir(parents=True, exist_ok=True)

        # Generate filename with timestamp
        filename = f"jobs_{collection_time.strftime('%H%M%S')}.json"
        filepath = source_dir / filename

        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✓ Saved {len(data)} jobs to {filepath}")

        return filepath
