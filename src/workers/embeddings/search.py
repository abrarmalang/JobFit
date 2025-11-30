import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from shared.config import get_settings
from sklearn.metrics.pairwise import cosine_similarity

class Search:
    def __init__(self, model_name='all-mpnet-base-v2'):
        self.config = get_settings()
        # A smaller, faster model might be better for on-the-fly CV embedding
        self.model = SentenceTransformer(model_name)
        self.jobs_df, self.embeddings = self._load_data(model_name)

    def _load_data(self, model_name):
        processed_path = self.config.data_dir / "processed"
        jobs_df = pd.read_parquet(os.path.join(processed_path, 'jobs.parquet'))

        # --- Data Augmentation & Cleaning ---
        # Calculate job age in days
        jobs_df['created_date'] = pd.to_datetime(jobs_df['created_date'], format='ISO8601', errors='coerce').dt.tz_localize(None)
        jobs_df['job_age_days'] = (pd.to_datetime('today') - jobs_df['created_date']).dt.days

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

        # Ensure numeric fields are numeric
        for col in ['salary_min', 'salary_max', 'job_age_days']:
            if col in jobs_df.columns:
                jobs_df[col] = pd.to_numeric(jobs_df[col], errors='coerce').fillna(0)


        # Ensure the embeddings file exists
        embeddings_dir = self.config.models_dir / "embeddings" / model_name
        # The project spec mentions parquet files for embeddings, but let's check for npy too
        embeddings_path_npy = os.path.join(embeddings_dir, 'jobs.npy')
        embeddings_path_parquet = os.path.join(embeddings_dir, 'jobs.parquet')

        if os.path.exists(embeddings_path_npy):
            embeddings = np.load(embeddings_path_npy)
        elif os.path.exists(embeddings_path_parquet):
            # Assuming the embedding is stored in a column named 'embedding'
            embeddings_df = pd.read_parquet(embeddings_path_parquet)
            embeddings = np.stack(embeddings_df['embedding_full'].to_numpy())
        else:
            # Handle case where embeddings are not found
            # For now, create zero embeddings as a fallback for initialization
            print(f"Warning: Embedding file not found for model {model_name}. Semantic search will not work.")
            embeddings = np.zeros((len(jobs_df), 1))

        return jobs_df, embeddings

    def keyword_search(self, query: str, location: str, skills: str, page: int = 1, page_size: int = 10):
        results_df = self.jobs_df.copy()
        
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

        return {
            "results": paginated_results.to_dict(orient='records'),
            "total_results": total_results
        }

    def semantic_search(self, query_text: str, top_n: int = 100, page: int = 1, page_size: int = 10):
        if self.embeddings.shape[1] == 1: # Check if we have real embeddings
             return {"results": [], "total_results": 0}

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
        
        return {
            "results": results_df.to_dict(orient='records'),
            "total_results": total_results
        }