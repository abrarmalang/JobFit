"""
Multi-Level Embedding Generator

Generates embeddings for job data following the AI-Native approach.
Creates multiple embeddings per job for different aspects:
- Full job description (overall context)
- Title + skills (role matching)
- Skills summary (semantic understanding)
"""

from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

PARQUET_WRITE_KWARGS = {
    "index": False,
    "compression": "zstd",
    "engine": "pyarrow",
    "use_dictionary": True
}

from .embedder_base import EmbedderBase
from .embedder_sentencetransformer import SentenceTransformerEmbedder

# Import metrics functions (optional)
try:
    from workers.metrics import (
        update_embedding_worker_status,
        update_model_status,
        log_embedding_run
    )
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False


class JobEmbeddingGenerator:
    """Generates multi-level embeddings for job data."""

    def __init__(
        self,
        embedder: EmbedderBase,
        data_dir: Path = None,
        models_dir: Path = None
    ):
        """
        Initialize embedding generator.

        Args:
            embedder: Embedding model to use
            data_dir: Root data directory
            models_dir: Root models directory
        """
        self.embedder = embedder

        # Set up directories
        if data_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            data_dir = project_root / "data"
        if models_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent
            models_dir = project_root / "models"

        self.data_dir = data_dir
        self.models_dir = models_dir
        self.processed_dir = data_dir / "processed"
        self.embeddings_dir = models_dir / "embeddings" / embedder.get_model_name()
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)

    def generate_embeddings(
        self,
        batch_size: int = 32,
        show_progress: bool = True
    ) -> Dict:
        """
        Generate embeddings for all jobs in processed dataset.

        Args:
            batch_size: Batch size for embedding generation
            show_progress: Show progress bar

        Returns:
            Statistics about the generation process
        """
        start_time = datetime.now()

        # Update worker and model status
        if METRICS_AVAILABLE:
            update_embedding_worker_status(
                "RUNNING",
                {"started_at": start_time.isoformat()}
            )
            update_model_status(
                model_name=self.embedder.get_model_name(),
                status="ONLINE",
                service_type="SentenceTransformer",
                embedding_dim=self.embedder.get_embedding_dim()
            )

        # Load processed jobs
        jobs_path = self.processed_dir / "jobs.parquet"
        if not jobs_path.exists():
            raise FileNotFoundError(
                f"Processed jobs file not found: {jobs_path}\n"
                "Run 'python scripts/run_processing.py' first to consolidate data."
            )

        print(f"\n{'='*70}")
        print(f"EMBEDDING GENERATION")
        print(f"{'='*70}")
        print(f"Model: {self.embedder.get_model_name()}")
        print(f"Embedding dim: {self.embedder.get_embedding_dim()}")
        print(f"Loading jobs from: {jobs_path}")

        df = pd.read_parquet(jobs_path)
        print(f"Loaded {len(df)} jobs")

        # Generate embeddings for each aspect
        print(f"\n{'─'*70}")
        print("Generating multi-level embeddings...")
        print(f"{'─'*70}\n")

        # 1. Full text embeddings (job description)
        print("[1/4] Full job description embeddings...")
        full_text_list = df['description'].fillna('').tolist()
        full_text_embeddings = self.embedder.encode(
            full_text_list,
            batch_size=batch_size,
            normalize=True,
            show_progress=show_progress
        )

        # 2. Title embeddings
        print("[2/4] Job title embeddings...")
        title_list = df['title'].fillna('').tolist()
        title_embeddings = self.embedder.encode(
            title_list,
            batch_size=batch_size,
            normalize=True,
            show_progress=show_progress
        )

        # 3. Title + Company embeddings (helps with company-specific roles)
        print("[3/4] Title + Company embeddings...")
        title_company_list = [
            f"{row['title']} at {row['company']}"
            for _, row in df.iterrows()
        ]
        title_company_embeddings = self.embedder.encode(
            title_company_list,
            batch_size=batch_size,
            normalize=True,
            show_progress=show_progress
        )

        # 4. Category + Title embeddings (helps with category-aware matching)
        print("[4/4] Category + Title embeddings...")
        category_title_list = [
            f"{row.get('category_label', '')} {row['title']}"
            for _, row in df.iterrows()
        ]
        category_title_embeddings = self.embedder.encode(
            category_title_list,
            batch_size=batch_size,
            normalize=True,
            show_progress=show_progress
        )

        # Create embeddings dataframe with job metadata
        embeddings_df = pd.DataFrame({
            'job_id': df['id'],
            'title': df['title'],
            'company': df['company'],
            'source': df['source'],
            'location_display': df['location_display'],
            'location_country': df.get('location_country', ''),
            'category_label': df.get('category_label', ''),
            'salary_min': df.get('salary_min', np.nan),
            'salary_max': df.get('salary_max', np.nan),
            'created_date': df.get('created_date', ''),
            'redirect_url': df['redirect_url'],

            # Embeddings as arrays
            'embedding_full': list(full_text_embeddings),
            'embedding_title': list(title_embeddings),
            'embedding_title_company': list(title_company_embeddings),
            'embedding_category_title': list(category_title_embeddings)
        })

        # Save embeddings dataframe
        output_path = self.embeddings_dir / "jobs.parquet"
        embeddings_df.to_parquet(output_path, **PARQUET_WRITE_KWARGS)

        # Create README
        self._create_readme(
            model_name=self.embedder.get_model_name(),
            embedding_dim=self.embedder.get_embedding_dim(),
            num_jobs=len(df),
            generation_time=datetime.now()
        )

        duration = (datetime.now() - start_time).total_seconds()

        # Statistics
        stats = {
            'status': 'success',
            'model_name': self.embedder.get_model_name(),
            'embedding_dim': self.embedder.get_embedding_dim(),
            'num_jobs': len(df),
            'num_embeddings_per_job': 4,
            'total_embeddings': len(df) * 4,
            'output_path': str(output_path),
            'duration_seconds': duration,
            'embeddings_per_second': (len(df) * 4) / duration if duration > 0 else 0,
            'timestamp': datetime.now().isoformat()
        }

        print(f"\n{'='*70}")
        print(f"GENERATION COMPLETE")
        print(f"{'='*70}")
        print(f"Jobs processed: {stats['num_jobs']}")
        print(f"Embeddings per job: {stats['num_embeddings_per_job']}")
        print(f"Total embeddings: {stats['total_embeddings']}")
        print(f"Duration: {stats['duration_seconds']:.2f}s")
        print(f"Speed: {stats['embeddings_per_second']:.1f} embeddings/sec")
        print(f"Output: {output_path}")
        print(f"{'='*70}\n")

        # Log metrics
        if METRICS_AVAILABLE:
            log_embedding_run(stats)
            update_embedding_worker_status(
                "IDLE",
                {
                    "last_run": stats['timestamp'],
                    "jobs_processed": stats['num_jobs']
                }
            )

        return stats

    def _create_readme(
        self,
        model_name: str,
        embedding_dim: int,
        num_jobs: int,
        generation_time: datetime
    ):
        """Create README file for embeddings directory."""
        readme_path = self.embeddings_dir / "README.md"

        content = f"""# Job Embeddings - {model_name}

## Model Information

- **Model Name:** {model_name}
- **Embedding Dimension:** {embedding_dim}
- **Generated:** {generation_time.strftime('%Y-%m-%d %H:%M:%S')}
- **Number of Jobs:** {num_jobs}

## Embedding Levels

This dataset contains multi-level embeddings for each job:

1. **embedding_full** - Full job description
   - Captures overall context and requirements
   - Use for: General semantic matching

2. **embedding_title** - Job title only
   - Captures role and seniority level
   - Use for: Role-specific matching

3. **embedding_title_company** - Title + Company
   - Captures company-specific role context
   - Use for: Company preference matching

4. **embedding_category_title** - Category + Title
   - Captures category-aware role matching
   - Use for: Cross-category exploration

## File Format

- **Format:** Apache Parquet
- **Compression:** Snappy (default)
- **Schema:**
  - `job_id` (string): Unique job identifier
  - `title` (string): Job title
  - `company` (string): Company name
  - `source` (string): Data source (adzuna, remoteok)
  - `location_display` (string): Location
  - `location_country` (string): Country code
  - `category_label` (string): Job category
  - `salary_min` (float): Minimum salary
  - `salary_max` (float): Maximum salary
  - `created_date` (string): Job posting date
  - `redirect_url` (string): Job URL
  - `embedding_full` (array): Full description embedding
  - `embedding_title` (array): Title embedding
  - `embedding_title_company` (array): Title+Company embedding
  - `embedding_category_title` (array): Category+Title embedding

## Usage

```python
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load embeddings
df = pd.read_parquet('jobs.parquet')

# Example: Find similar jobs by title
query_embedding = df.iloc[0]['embedding_title']
similarities = cosine_similarity(
    [query_embedding],
    list(df['embedding_title'])
)[0]

# Get top 10 most similar
top_indices = similarities.argsort()[-10:][::-1]
similar_jobs = df.iloc[top_indices]
```

## Matching Strategy

For CV-to-job matching, use weighted combination:

- 20% full description match (overall context)
- 35% skill-focused match (will be added with skill extraction)
- 30% required skills match (will be added with skill extraction)
- 15% semantic summary match (will be added with skill extraction)

For now, use title and full description embeddings as baseline.
"""

        with open(readme_path, 'w') as f:
            f.write(content)
