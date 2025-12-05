"""
Adzuna API Collector

Fetches job postings from Adzuna API with deduplication and date filtering.
Uses multithreading to speed up page fetching.

Compatible with:
- main_collection.py (AdzunaCollector(app_id=..., app_key=...))
- CollectorBase (fetch_jobs, _structure_job_data, get_statistics)

Enhanced with:
- Auto country injection per job
- Auto category inference from title/description when missing
"""

import os
import requests
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

from .collector_base import CollectorBase


class AdzunaCollector(CollectorBase):
    """
    Collector implementation for the Adzuna jobs API.

    main_collection.py expects:
        collector = AdzunaCollector(app_id=..., app_key=...)
        result = collector.fetch_jobs(...)
        stats  = collector.get_statistics()
    """

    BASE_URL = "https://api.adzuna.com/v1/api/jobs/{country}/search/{page}"

    # Simple mapping from Adzuna country codes to human-friendly names
    COUNTRY_LABELS = {
        "gb": "UK",
        "us": "USA",
        "ca": "Canada",
        "au": "Australia",
        "in": "India",
        "sg": "Singapore",
        "za": "South Africa",
        "fr": "France",
        "de": "Germany",
        "nz": "New Zealand",
        "ie": "Ireland",
    }

    def __init__(
        self,
        app_id: str,
        app_key: str,
        data_dir: Optional[Path] = None,
        max_pages: int = 200,
        max_workers: int = 3,
        batch_save_size: int = 500,
    ):
        """
        Args:
            app_id: Adzuna APP ID
            app_key: Adzuna APP KEY
            data_dir: Optional root data directory (CollectorBase will default)
            max_pages: max pages per query (safety cap)
            max_workers: number of threads for parallel page fetching
            batch_save_size: Number of jobs to collect before saving (0 = save at end)
        """
        super().__init__(data_dir=data_dir)

        # Prefer explicit values, fallback to env if needed
        self.app_id = app_id or os.getenv("ADZUNA_APP_ID")
        self.app_key = app_key or os.getenv("ADZUNA_APP_KEY")

        if not self.app_id or not self.app_key:
            raise ValueError(
                "AdzunaCollector requires app_id and app_key "
                "(or ADZUNA_APP_ID/ADZUNA_APP_KEY in env)."
            )

        self.max_pages = max_pages
        self.max_workers = max_workers
        self.batch_save_size = batch_save_size

        # Will hold stats from the last fetch_jobs run
        self._last_stats: Dict[str, Any] = {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }

        # Used to inject country info into each job
        self._current_country_code: Optional[str] = None

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _save_batch(
        self,
        jobs: List[Dict],
        saved_files: List[Path],
        batch_num: int,
        is_final: bool
    ) -> int:
        """
        Save a batch of jobs to disk and track the file.

        Args:
            jobs: List of job dictionaries to save
            saved_files: List to append the saved file path to
            batch_num: Batch number for logging
            is_final: Whether this is the final batch

        Returns:
            Number of jobs saved
        """
        filepath = self.save_raw_list(jobs, "adzuna", silent=True)
        saved_files.append(filepath)

        batch_label = "Final batch" if is_final else f"Batch {batch_num}"
        print(f"[Adzuna] {batch_label}: {len(jobs)} jobs saved -> {filepath.name}")

        return len(jobs)

    def _build_params(
        self,
        what: Optional[str],
        where: Optional[str],
        category: Optional[str],
        max_days_old: int,
        results_per_page: int,
    ) -> Dict[str, Any]:
        """Build query parameters for Adzuna API."""
        params: Dict[str, Any] = {
            "app_id": self.app_id,
            "app_key": self.app_key,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        if what:
            params["what"] = what
        if where:
            params["where"] = where
        if category:
            params["category"] = category

        if max_days_old is not None:
            params["max_days_old"] = max_days_old

        return params

    def _fetch_page(
        self,
        country: str,
        page: int,
        base_params: Dict[str, Any],
    ) -> List[Dict]:
        """Fetch a single page from Adzuna with timeout and logging."""
        url = self.BASE_URL.format(country=country, page=page)
        print(f"[Adzuna]   → Requesting page {page} for country={country}...")

        try:
            response = requests.get(url, params=base_params, timeout=20)
        except requests.RequestException as e:
            print(f"[Adzuna] Request exception on page {page}: {e}")
            return []

        if response.status_code == 429:
            print(f"[Adzuna] Rate limited (429) at page {page}")
            return []

        if response.status_code != 200:
            print(f"[Adzuna] Request failed: {response.status_code} at page {page}")
            return []

        data = response.json()
        results = data.get("results", [])
        print(f"[Adzuna]   ← Page {page} returned {len(results)} jobs")

        return results

    @staticmethod
    def _infer_category_from_text(
        title: Optional[str],
        description: Optional[str],
        raw_category: Optional[str],
    ) -> str:
        """
        Infer a coarse job category from title/description when Adzuna category is missing.

        This is a simple keyword-based classifier, good enough for analytics + clustering.
        """
        if raw_category:
            # Use Adzuna's own category label when available
            return raw_category

        text = f"{title or ''} {description or ''}".lower()

        # IT / Engineering / Data
        if any(
            kw in text
            for kw in [
                "software engineer",
                "developer",
                "frontend",
                "backend",
                "full stack",
                "devops",
                "data scientist",
                "data engineer",
                "machine learning",
                "ml engineer",
                "python",
                "java",
                "golang",
                "c++",
                "react",
                "typescript",
                "cloud",
                "aws",
                "azure",
                "kubernetes",
            ]
        ):
            return "IT & Engineering"

        # Data / Analytics
        if any(
            kw in text
            for kw in [
                "data analyst",
                "business intelligence",
                "analytics",
                "bi analyst",
                "statistician",
            ]
        ):
            return "Data & Analytics"

        # Finance / Accounting
        if any(
            kw in text
            for kw in [
                "accountant",
                "finance",
                "financial analyst",
                "auditor",
                "tax",
                "bookkeeper",
                "controller",
                "fund",
                "banking",
            ]
        ):
            return "Finance & Accounting"

        # Marketing / Content
        if any(
            kw in text
            for kw in [
                "marketing",
                "digital marketing",
                "seo",
                "sem",
                "social media",
                "content writer",
                "copywriter",
                "brand",
                "growth marketer",
            ]
        ):
            return "Marketing & Content"

        # Sales / Business
        if any(
            kw in text
            for kw in [
                "sales",
                "account executive",
                "business development",
                "bdr",
                "sdr",
                "inside sales",
                "sales manager",
                "sales representative",
            ]
        ):
            return "Sales & Business Development"

        # HR / People
        if any(
            kw in text
            for kw in [
                "human resources",
                "hr manager",
                "recruiter",
                "talent acquisition",
                "people ops",
            ]
        ):
            return "HR & Recruitment"

        # Health / Medical
        if any(
            kw in text
            for kw in [
                "nurse",
                "doctor",
                "gp",
                "physician",
                "medical",
                "clinical",
                "hospital",
                "healthcare",
                "care assistant",
                "carer",
            ]
        ):
            return "Healthcare & Nursing"

        # Education / Teaching
        if any(
            kw in text
            for kw in [
                "teacher",
                "teaching assistant",
                "lecturer",
                "professor",
                "tutor",
                "education",
                "school",
                "academy",
            ]
        ):
            return "Education & Teaching"

        # Design / Creative
        if any(
            kw in text
            for kw in [
                "ux",
                "ui",
                "designer",
                "graphic design",
                "illustrator",
                "creative",
                "product design",
                "animation",
                "3d artist",
            ]
        ):
            return "Design & Creative"

        # Operations / Logistics
        if any(
            kw in text
            for kw in [
                "operations",
                "logistics",
                "supply chain",
                "warehouse",
                "procurement",
                "fleet",
            ]
        ):
            return "Operations & Logistics"

        # Management / PM
        if any(
            kw in text
            for kw in [
                "project manager",
                "programme manager",
                "product manager",
                "scrum master",
                "delivery manager",
            ]
        ):
            return "Project & Product Management"

        # Default
        return "Unknown"

    def _resolve_country(self, raw_job: Dict) -> str:
        """
        Decide the country label for a job.

        Priority:
        1) Use the country code passed to fetch_jobs (self._current_country_code)
        2) Try Adzuna location.area[0]
        3) Fallback to 'Unknown'
        """
        # 1) From current fetch context
        if self._current_country_code:
            code = self._current_country_code.lower()
            return self.COUNTRY_LABELS.get(code, code.upper())

        # 2) From job location area, if present
        area = raw_job.get("location", {}).get("area") or []
        if area:
            # Often looks like ["UK", "London", ...]
            return str(area[0])

        # 3) Fallback
        return "Unknown"

    # ------------------------------------------------------------------ #
    # Required interface for CollectorBase / main_collection
    # ------------------------------------------------------------------ #

    def fetch_jobs(
        self,
        country: str,
        what: Optional[str],
        where: Optional[str],
        category: Optional[str],
        max_days_old: int,
        max_jobs: int,
        results_per_page: int,
    ) -> Dict:
        """
        Fetch jobs from Adzuna using provided filters.

        This matches how main_collection.py calls it.

        Uses multithreading to fetch multiple pages in parallel.
        """
        print("\n================ ADZUNA COLLECTION ====================")
        print(f"  Country        : {country}")
        print(f"  What (keyword) : {what}")
        print(f"  Where (location): {where}")
        print(f"  Category       : {category}")
        print(f"  Max days old   : {max_days_old}")
        print(f"  Max jobs       : {max_jobs}")
        print(f"  Results / page : {results_per_page}")
        print(f"  Max pages      : {self.max_pages}")
        print(f"  Max workers    : {self.max_workers}")
        print("=======================================================\n")

        # Remember which country this run is for
        self._current_country_code = country

        start_time = datetime.now()   # for duration
        collected_jobs: List[Dict] = []
        seen_ids: Set[str] = set()
        saved_files: List[Path] = []  # Track all saved batch files

        total_api_results = 0
        api_calls = 0
        collection_time = datetime.now()

        base_params = self._build_params(
            what=what,
            where=where,
            category=category,
            max_days_old=max_days_old,
            results_per_page=results_per_page,
        )

        page = 1
        batch_size = self.max_workers  # number of pages fetched in parallel
        total_jobs_saved = 0  # Track total across all saved batches

        while total_jobs_saved + len(collected_jobs) < max_jobs and page <= self.max_pages:
            pages_batch = list(range(page, min(page + batch_size, self.max_pages + 1)))
            if not pages_batch:
                break

            print(f"[Adzuna] Fetching pages {pages_batch[0]}-{pages_batch[-1]}...")

            # Fetch pages in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {
                    executor.submit(self._fetch_page, country, p, base_params): p
                    for p in pages_batch
                }
                api_calls += len(futures)

                # Process results as they complete (streaming)
                for future in as_completed(futures):
                    results = future.result() or []
                    total_api_results += len(results)

                    # Process jobs immediately (avoid storing raw results)
                    for raw_job in results:
                        # Early filtering before processing
                        job_id = str(raw_job.get("id") or raw_job.get("adref") or "")
                        if not job_id or job_id in seen_ids:
                            continue

                        # Date filter (early exit)
                        if max_days_old is not None:
                            created = raw_job.get("created")
                            if created:
                                try:
                                    created_dt = datetime.fromisoformat(
                                        created.replace("Z", "+00:00")
                                    ).replace(tzinfo=None)
                                    if created_dt < datetime.now() - timedelta(days=max_days_old):
                                        continue
                                except Exception:
                                    pass

                        # Structure and collect
                        structured = self._structure_job_data(raw_job)
                        collected_jobs.append(structured)
                        seen_ids.add(job_id)

                        # Check if we hit max_jobs
                        if total_jobs_saved + len(collected_jobs) >= max_jobs:
                            break

            # Early exit if no results from entire batch
            if api_calls > 0 and total_api_results == 0:
                print("[Adzuna] No results returned. Stopping.")
                break

            # Incremental save if batch_save_size is set and reached
            if self.batch_save_size > 0 and len(collected_jobs) >= self.batch_save_size:
                total_jobs_saved += self._save_batch(
                    collected_jobs[:self.batch_save_size],
                    saved_files,
                    batch_num=len(saved_files) + 1,
                    is_final=False
                )
                collected_jobs = collected_jobs[self.batch_save_size:]

            page += batch_size

        end_time = datetime.now()
        duration_seconds = (end_time - start_time).total_seconds()

        # Save any remaining jobs
        if len(collected_jobs) > 0:
            total_jobs_saved += self._save_batch(
                collected_jobs,
                saved_files,
                batch_num=len(saved_files) + 1,
                is_final=True
            )

        # Total jobs collected
        total_jobs_collected = total_jobs_saved

        print(
            f"\n[Adzuna] Finished collection: {total_jobs_collected} total jobs "
            f"(from {total_api_results} API results, {len(saved_files)} files)."
        )

        # Update last_stats so get_statistics() can report correctly
        last_filepath = saved_files[-1] if saved_files else None
        self._last_stats = {
            "unique_jobs": total_jobs_collected,
            "total_collection_files": len(saved_files),
            "raw_data_directory": str(Path(last_filepath).parent) if last_filepath else "",
        }

        print("\n================ ADZUNA COLLECTION SUMMARY ============")
        print(f"  Jobs collected      : {total_jobs_collected}")
        print(f"  Approx. API results : {total_api_results}")
        print(f"  API calls attempted : {api_calls}")
        print(f"  Duration (seconds)  : {duration_seconds:.2f}")
        print(f"  Files saved         : {len(saved_files)}")
        if saved_files:
            print(f"  Output directory    : {Path(saved_files[0]).parent}")
        print("=======================================================\n")

        # status expected by main_collection:
        # - "success" when we collected jobs
        # - "no_new_jobs" when 0 collected
        status = "success" if total_jobs_collected > 0 else "no_new_jobs"

        # Clear current country context
        self._current_country_code = None

        return {
            "timestamp": collection_time.isoformat(),
            "jobs_collected": total_jobs_collected,
            "jobs_deduplicated": total_jobs_collected,
            "filepath": str(last_filepath) if last_filepath else "",
            "source": "adzuna",
            "api_calls": api_calls,
            "duration_seconds": duration_seconds,
            "status": status,
        }

    def _structure_job_data(self, raw_job: Dict) -> Dict:
        """
        Convert raw Adzuna job JSON into a standardized format.

        Enhanced with:
        - category: inferred from title/description if Adzuna category is missing
        - Maintains compatibility with original field names for downstream pipeline
        """
        # Extract raw fields
        title = raw_job.get("title")
        description = raw_job.get("description", "")
        raw_category_label = (raw_job.get("category") or {}).get("label")
        raw_category_tag = (raw_job.get("category") or {}).get("tag")

        # Parse location hierarchy
        location = raw_job.get("location", {})
        location_area = location.get("area", [])

        # Calculate job age
        created_date = raw_job.get("created")
        job_age_days = None
        if created_date:
            try:
                created = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                job_age_days = (datetime.now(created.tzinfo) - created).days
            except:
                pass

        # Use raw category if available, otherwise infer
        category_label = raw_category_label or self._infer_category_from_text(
            title=title,
            description=description,
            raw_category=raw_category_label,
        )

        return {
            # Core fields - use original field names
            "id": str(raw_job.get("id")),
            "title": title,
            "company": raw_job.get("company", {}).get("display_name", "Unknown"),
            "description": description,
            "redirect_url": raw_job.get("redirect_url"),

            # Category - maintain original field names
            "category_tag": raw_category_tag,
            "category_label": category_label,

            # Location - maintain original hierarchy
            "location_display": location.get("display_name"),
            "location_country": location_area[0] if len(location_area) > 0 else None,
            "location_region": location_area[1] if len(location_area) > 1 else None,
            "location_city": location_area[2] if len(location_area) > 2 else None,
            "latitude": raw_job.get("latitude"),
            "longitude": raw_job.get("longitude"),

            # Salary
            "salary_min": raw_job.get("salary_min"),
            "salary_max": raw_job.get("salary_max"),
            "salary_is_predicted": raw_job.get("salary_is_predicted", "1") == "0",

            # Employment
            "contract_type": raw_job.get("contract_type"),
            "contract_time": raw_job.get("contract_time"),
            "created_date": created_date,
            "job_age_days": job_age_days,

            # Metadata
            "source": "adzuna",
            "collected_at": datetime.now().isoformat(),
        }

    def get_statistics(self) -> Dict:
        """
        Return statistics for the last collection run.

        main_collection.py expects:
            stats["unique_jobs"]
            stats["total_collection_files"]
            stats["raw_data_directory"]
        """
        return self._last_stats or {
            "unique_jobs": 0,
            "total_collection_files": 0,
            "raw_data_directory": "",
        }
