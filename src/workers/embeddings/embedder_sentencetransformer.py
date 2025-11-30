"""
Sentence Transformer Embedder

Local embedding model using sentence-transformers library.
Supports models like all-mpnet-base-v2, multilingual-e5-large, etc.
"""

from typing import List, Union, Optional
import numpy as np
from pathlib import Path

from .embedder_base import EmbedderBase


class SentenceTransformerEmbedder(EmbedderBase):
    """Embedder using sentence-transformers for local embedding generation."""

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        cache_dir: Optional[str] = None
    ):
        """
        Initialize sentence transformer embedder.

        Args:
            model_name: Name of the sentence-transformer model
            device: Device to use (cpu, cuda, mps)
            cache_dir: Directory to cache downloaded models
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers"
            )

        self.model_name = model_name
        self.device = device

        # Load model
        print(f"Loading embedding model: {model_name} on {device}...")
        self.model = SentenceTransformer(
            model_name,
            device=device,
            cache_folder=cache_dir
        )
        print(f"Model loaded successfully. Embedding dim: {self.get_embedding_dim()}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings using sentence-transformers.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        # Handle single text
        if isinstance(texts, str):
            texts = [texts]

        # Encode
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )

        return embeddings

    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this model."""
        return self.model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get the name/identifier of the embedding model."""
        return self.model_name
