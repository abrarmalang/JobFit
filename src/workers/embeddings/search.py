import logging
import os
import platform
try:
    import resource  # Unix only
except ImportError:  # pragma: no cover
    resource = None
import pandas as pd
import numpy as np
from shared.config import get_settings

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None

try:
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover
    cosine_similarity = None

try:
    import psutil  # Optional, for accurate cross-platform RSS readings
except ImportError:  # pragma: no cover - psutil isn't guaranteed to exist
    psutil = None


logger = logging.getLogger(__name__)


class Search:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.config = get_settings()
        self.model_name = model_name
        self.semantic_available = SentenceTransformer is not None and cosine_similarity is not None
        self.model = None  # Lazily loaded
        self.embeddings = None  # Lazily loaded, potentially memory-mapped
        self.jobs_df = self._load_jobs()
        self._log_memory("jobs-loaded")

    def _current_memory_mb(self):
        """Return RSS in MB, using psutil when available."""
        if psutil is not None:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)

        if resource is not None:
            usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            if platform.system() == "Darwin":
                return usage / (1024 * 1024)
            # Linux reports kilobytes
            return usage / 1024

        return float("nan")

    def _log_memory(self, stage: str):
        mem_mb = self._current_memory_mb()
        message = f"[Search] {stage} memory usage: {mem_mb:.2f} MB"
        print(message)
        logger.info(message)

    def _load_jobs(self):
        processed_path = self.config.data_dir / "processed"
        jobs_df = pd.read_parquet(os.path.join(processed_path, 'jobs.parquet'))

        # Convert to narrower dtypes to reduce resident memory
        dtype_overrides = {
            "salary_min": "float32",
            "salary_max": "float32"
        }
        for col, dtype in dtype_overrides.items():
            if col in jobs_df.columns:
                jobs_df[col] = pd.to_numeric(jobs_df[col], errors='coerce').fillna(0).astype(dtype)

        # --- Data Augmentation & Cleaning ---
        # Calculate job age in days
        jobs_df['created_date'] = pd.to_datetime(jobs_df['created_date'], format='ISO8601', errors='coerce').dt.tz_localize(None)
        days_delta = (pd.Timestamp.utcnow().normalize() - jobs_df['created_date']).dt.days
        jobs_df['job_age_days'] = days_delta.fillna(0).astype('int16')

        # Add placeholder for contract_time if it doesn't exist
        if 'contract_time' not in jobs_df.columns:
            jobs_df['contract_time'] = 'N/A'
        
        # Fill missing descriptions
        if 'description' not in jobs_df.columns:
            jobs_df['description'] = 'No description available.'
        else:
            jobs_df['description'] = jobs_df['description'].fillna('No description available.')

        # Ensure other critical text fields are not null
        for col in ['title', 'company', 'location_display', 'salary_min', 'salary_max', 'redirect_url']:
            if col in jobs_df.columns:
                jobs_df[col] = jobs_df[col].fillna('Not specified')
            else:
                jobs_df[col] = 'Not specified'

        # Convert string-heavy columns to categorical to shrink in-memory size
        categorical_cols = ['title', 'company', 'location_display', 'contract_time', 'redirect_url']
        for col in categorical_cols:
            if col in jobs_df.columns:
                jobs_df[col] = jobs_df[col].astype('category')

        return jobs_df

    def _ensure_model_loaded(self):
        if not self.semantic_available:
            return
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)

    def _ensure_embeddings_loaded(self):
        if not self.semantic_available:
            return
        if self.embeddings is not None:
            return

        jobs_count = len(self.jobs_df)
        embeddings_dir = self.config.models_dir / "embeddings" / self.model_name
        embeddings_path_npy = os.path.join(embeddings_dir, 'jobs.npy')
        embeddings_path_parquet = os.path.join(embeddings_dir, 'jobs.parquet')

        if os.path.exists(embeddings_path_npy):
            # Use memory-mapped mode so the embeddings stay off-heap until accessed
            self.embeddings = np.load(embeddings_path_npy, mmap_mode='r')
            return

        if os.path.exists(embeddings_path_parquet):
            embeddings_df = pd.read_parquet(embeddings_path_parquet)
            self.embeddings = np.stack(embeddings_df['embedding_full'].to_numpy())
            return

        print(f"Warning: Embedding file not found for model {self.model_name}. Semantic search will not work.")
        self.embeddings = np.zeros((jobs_count, 1))

    def keyword_search(self, query: str, location: str, skills: str, page: int = 1, page_size: int = 10):
        self._log_memory("keyword:start")
        results_df = self.jobs_df
        
        if query:
            results_df = results_df[results_df['title'].str.contains(query, case=False, na=False) |
                                    results_df['description'].str.contains(query, case=False, na=False)]
        
        if location:
            results_df = results_df[results_df['location_display'].str.contains(location, case=False, na=False)]
            
        if skills:
            for skill in skills.split(','):
                skill = skill.strip()
                if skill:
                    results_df = results_df[results_df['description'].str.contains(skill, case=False, na=False)]

        total_results = len(results_df)
        
        # Paginate results
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_results = results_df.iloc[start_index:end_index].copy()

        # Replace NaN with None for JSON compatibility
        paginated_results = paginated_results.replace({np.nan: None})

        # Convert all object columns to strings to ensure serializability
        for col in paginated_results.columns:
            if paginated_results[col].dtype == 'object':
                paginated_results[col] = paginated_results[col].astype(str)

        # Convert datetime objects to strings
        if 'created_date' in paginated_results.columns:
            paginated_results['created_date'] = paginated_results['created_date'].astype(str)

        payload = {
            "results": paginated_results.to_dict(orient='records'),
            "total_results": total_results
        }
        self._log_memory("keyword:end")
        return payload

    def semantic_search(self, query_text: str, top_n: int = 100, page: int = 1, page_size: int = 10):
        if not self.semantic_available:
            self._log_memory("semantic:disabled")
            raise RuntimeError("Semantic search is not enabled on this deployment. Install worker requirements.")
        self._log_memory("semantic:init")
        self._ensure_model_loaded()
        self._ensure_embeddings_loaded()
        self._log_memory("semantic:embeddings-loaded")

        if self.embeddings.shape[1] == 1:  # Check if we have real embeddings
            payload = {"results": [], "total_results": 0}
            self._log_memory("semantic:no-embeddings")
            return payload

        query_embedding = self.model.encode([query_text])

        similarities = cosine_similarity(query_embedding, self.embeddings).flatten()
        
        # Get top N results, then paginate
        # This is more efficient than paginating the entire dataset first
        top_indices = np.argsort(similarities)[-top_n:][::-1]
        
        total_results = len(top_indices)

        # Paginate the top N indices
        start_index = (page - 1) * page_size
        end_index = start_index + page_size
        paginated_indices = top_indices[start_index:end_index]

        if len(paginated_indices) == 0:
            return {"results": [], "total_results": total_results}

        results_df = self.jobs_df.iloc[paginated_indices].copy()
        results_df['match_score'] = [round(s * 100) for s in similarities[paginated_indices]]
        
        # Replace NaN with None for JSON compatibility
        results_df = results_df.replace({np.nan: None})
        
        # Convert all object columns to strings to ensure serializability
        for col in results_df.columns:
            if results_df[col].dtype == 'object':
                results_df[col] = results_df[col].astype(str)

        # Convert datetime objects to strings
        if 'created_date' in results_df.columns:
            results_df['created_date'] = results_df['created_date'].astype(str)
        
        payload = {
            "results": results_df.to_dict(orient='records'),
            "total_results": total_results
        }
        self._log_memory("semantic:end")
        return payload
