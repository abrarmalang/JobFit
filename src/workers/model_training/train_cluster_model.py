#!/usr/bin/env python
"""
Train Job Clustering Model

This script:
  - Loads job embeddings from: models/embeddings/all-mpnet-base-v2/jobs.parquet
  - Auto-detects embedding structure (flexible with column naming)
  - Trains a KMeans clustering model
  - Saves:
      - clustered jobs table
      - trained model
      - simple metrics for reporting

Usage:
    python src/workers/model_training/train_cluster_model.py
"""

import os
from pathlib import Path
from typing import Tuple, Any, List

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from datetime import datetime


class JobClusterTrainer:
    def __init__(
        self,
        embeddings_path: Path = Path("models/embeddings/all-mpnet-base-v2/jobs.parquet"),
        output_dir: Path = Path("models/trained"),
        n_clusters: int = 20,
        random_state: int = 42,
    ):
        self.embeddings_path = embeddings_path
        self.output_dir = output_dir
        self.n_clusters = n_clusters
        self.random_state = random_state

        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- #
    # Loading & feature extraction
    # ------------------------------------------------------------- #

    def _detect_embedding_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Try multiple strategies to extract an embedding matrix from the DataFrame.

        Priority:
          1) Column named 'embedding', 'embeddings', or 'vector', where each cell is a list/array
          2) Any object column with list/array-like values of length > 10
          3) Fallback: all numeric columns as a feature matrix
        """
        print("🔍 Columns in embeddings file:")
        for col in df.columns:
            print(f"   - {col} (dtype={df[col].dtype})")

        # 1) Direct known column names
        candidate_names = ["embedding", "embeddings", "vector"]
        for name in candidate_names:
            if name in df.columns:
                print(f"\n✅ Found candidate embedding column: '{name}'")
                series = df[name].dropna()
                if not len(series):
                    continue
                first_val = series.iloc[0]
                if isinstance(first_val, (list, np.ndarray)):
                    X = np.vstack(series.apply(lambda x: np.array(x, dtype="float32")))
                    print(f"   → Using '{name}' as embedding matrix with shape {X.shape}")
                    return X

        # 2) Any object column with list/array-like content
        for col in df.columns:
            if df[col].dtype == "object":
                series = df[col].dropna()
                if not len(series):
                    continue
                first_val = series.iloc[0]
                if isinstance(first_val, (list, np.ndarray)) and len(first_val) > 10:
                    print(f"\n✅ Found list/array-like column '{col}' to use as embeddings")
                    X = np.vstack(series.apply(lambda x: np.array(x, dtype="float32")))
                    print(f"   → Using '{col}' as embedding matrix with shape {X.shape}")
                    return X

        # 3) Fallback: all numeric columns
        numeric_df = df.select_dtypes(include=["float32", "float64", "int32", "int64"])
        if numeric_df.shape[1] > 0:
            X = numeric_df.to_numpy(dtype="float32")
            print(
                f"\n⚠️ No explicit embedding column found. "
                f"Using all numeric columns as features, shape={X.shape}"
            )
            return X

        raise RuntimeError(
            "❌ Could not detect an embedding matrix. "
            "No suitable embedding or numeric columns found."
        )

    def load_embeddings(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Load the embeddings parquet and extract feature matrix."""
        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"❌ Embeddings file not found: {self.embeddings_path}\n"
                "   Make sure you ran: python scripts/run_generate_embeddings.py"
            )

        print(f"\n📄 Loading embeddings from: {self.embeddings_path}")
        df = pd.read_parquet(self.embeddings_path)
        print(f"   → Loaded {len(df)} rows and {len(df.columns)} columns")

        X = self._detect_embedding_matrix(df)

        if X.shape[0] != len(df):
            raise RuntimeError(
                f"❌ Mismatch between number of rows in DataFrame ({len(df)}) "
                f"and embedding matrix ({X.shape[0]})"
            )

        return df, X

    # ------------------------------------------------------------- #
    # Training
    # ------------------------------------------------------------- #

    def train_model(self, X: np.ndarray) -> Tuple[Any, dict]:
        """Train a KMeans clustering model and compute basic metrics."""
        n_samples = X.shape[0]
        n_clusters = min(self.n_clusters, max(2, n_samples // 300))
        if n_clusters < 2:
            n_clusters = 2

        print(f"\n🧠 Training KMeans clustering model...")
        print(f"   → Samples: {n_samples}")
        print(f"   → Desired clusters: {self.n_clusters} (adjusted to {n_clusters})")

        model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init="auto",
        )
        labels = model.fit_predict(X)

        inertia = model.inertia_
        sil_score = None
        if n_samples >= n_clusters * 2:
            try:
                sil_score = float(silhouette_score(X, labels))
            except Exception:
                sil_score = None

        unique, counts = np.unique(labels, return_counts=True)
        cluster_sizes = dict(zip(map(int, unique), map(int, counts)))

        print("\n📊 Clustering metrics:")
        print(f"   → Inertia: {inertia:.2f}")
        if sil_score is not None:
            print(f"   → Silhouette score: {sil_score:.4f}")
        print(f"   → Cluster sizes:")
        for cid, size in cluster_sizes.items():
            print(f"      - Cluster {cid}: {size} jobs")

        metrics = {
            "n_samples": int(n_samples),
            "n_clusters": int(n_clusters),
            "inertia": float(inertia),
            "silhouette_score": sil_score,
            "cluster_sizes": cluster_sizes,
        }

        return (model, labels, metrics)

    # ------------------------------------------------------------- #
    # Saving artifacts
    # ------------------------------------------------------------- #

    def save_artifacts(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
        model: Any,
        metrics: dict,
    ) -> None:
        """Save clustered data, model, and metrics to disk with fixed filenames."""
        # 1. Save clustered jobs table
        df_clustered = df.copy()
        df_clustered["cluster_id"] = labels.astype(int)
        clustered_path = self.output_dir / "job_clusters.parquet"
        df_clustered.to_parquet(clustered_path, index=False)

        # 2. Save model
        model_path = self.output_dir / "job_cluster_model.pkl"
        joblib.dump(model, model_path)

        # 3. Save metrics to common metrics folder
        metrics_dir = Path("models/metrics")
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Add timestamp to metrics
        metrics["trained_at"] = datetime.now().isoformat()

        metrics_path = metrics_dir / "training_metrics.json"
        pd.Series(metrics).to_json(metrics_path, indent=2)

        print("\n💾 Saved artifacts:")
        print(f"   → Clustered data : {clustered_path}")
        print(f"   → Model          : {model_path}")
        print(f"   → Metrics        : {metrics_path}")

    # ------------------------------------------------------------- #
    # Orchestration
    # ------------------------------------------------------------- #

    def run(self) -> None:
        """Full training pipeline."""
        df, X = self.load_embeddings()
        model, labels, metrics = self.train_model(X)
        self.save_artifacts(df, labels, model, metrics)

        print("\n✅ Training complete!")
        print(f"   Jobs clustered: {metrics['n_samples']}")
        print(f"   Clusters      : {metrics['n_clusters']}")
        print("   You can now integrate this model into your job recommender.")


if __name__ == "__main__":
    trainer = JobClusterTrainer()
    trainer.run()
