"""
Configuration Management

Type-safe configuration using Pydantic Settings.
Reads from .env file and environment variables.
"""

from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AdzunaConfig(BaseSettings):
    """Adzuna API configuration."""

    app_id: str = Field(default="", description="Adzuna API application ID")
    app_key: str = Field(default="", description="Adzuna API application key")

    # Collection settings
    country: str = Field(default="gb", description="Country code (gb, us, au, etc.)")
    what: str = Field(default="", description="Job search keywords")
    where: str = Field(default="", description="Location filter")
    category: Optional[str] = Field(default="it-jobs", description="Job category")
    max_days_old: int = Field(default=7, description="Only fetch jobs from last N days")
    max_jobs: int = Field(default=100, description="Maximum jobs to fetch per run")
    results_per_page: int = Field(default=50, description="Results per API call (max 50)")

    model_config = SettingsConfigDict(
        env_prefix="ADZUNA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class RemoteOKConfig(BaseSettings):
    """RemoteOK API configuration."""

    max_jobs: int = Field(default=100, description="Maximum jobs to fetch per run")
    filter_tags: Optional[List[str]] = Field(
        default=["python", "react", "javascript", "engineer"],
        description="Filter by tags (skills/technologies)"
    )

    model_config = SettingsConfigDict(
        env_prefix="REMOTEOK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class DataConfig(BaseSettings):
    """Data processing configuration."""

    output_format: str = Field(default="parquet", description="Output format (parquet or csv)")

    model_config = SettingsConfigDict(
        env_prefix="DATA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


class EmbeddingConfig(BaseSettings):
    """Embedding model configuration."""

    model_name: str = Field(
        default="all-mpnet-base-v2",
        description="Sentence transformer model name"
    )
    device: str = Field(
        default="cpu",
        description="Device to use (cpu, cuda, mps)"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Batch size for embedding generation"
    )
    normalize: bool = Field(
        default=True,
        description="Normalize embeddings to unit length"
    )
    cache_dir: Optional[str] = Field(
        default=None,
        description="Model cache directory"
    )

    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=()  # Allow model_* field names
    )


class Settings(BaseSettings):
    """Main application settings."""

    # Sub-configurations
    adzuna: AdzunaConfig = Field(default_factory=AdzunaConfig)
    remoteok: RemoteOKConfig = Field(default_factory=RemoteOKConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)

    # Project paths
    project_root: Path = Field(default_factory=lambda: Path.cwd())

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """Get models directory path."""
        return self.project_root / "models"


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get settings singleton instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


# Convenience function
def load_config() -> Settings:
    """Load configuration from .env and environment variables."""
    return get_settings()
