"""
Base Embedder Interface

Defines the pluggable architecture for embedding models.
Allows swapping between different providers (local, OpenAI, Cohere, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np


class EmbedderBase(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode texts into embeddings.

        Args:
            texts: Single text or list of texts to encode
            batch_size: Batch size for encoding
            normalize: Whether to normalize embeddings to unit length
            show_progress: Whether to show progress bar

        Returns:
            Numpy array of embeddings (shape: [n_texts, embedding_dim])
        """
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimensionality of embeddings produced by this model."""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name/identifier of the embedding model."""
        pass

    def encode_single(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Convenience method to encode a single text.

        Args:
            text: Text to encode
            normalize: Whether to normalize embedding

        Returns:
            1D numpy array of embedding
        """
        result = self.encode([text], batch_size=1, normalize=normalize)
        return result[0]
