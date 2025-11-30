"""
Generate Embeddings Script

Generates multi-level embeddings for all jobs in the processed dataset.
Run with: python scripts/run_generate_embeddings.py
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from shared.config import load_config
from workers.embeddings.embedder_sentencetransformer import SentenceTransformerEmbedder
from workers.embeddings.generator import JobEmbeddingGenerator


def main():
    """Generate embeddings for all jobs."""

    # Load configuration
    config = load_config()
    embedding_config = config.embedding

    print(f"\n{'='*70}")
    print(f"EMBEDDING GENERATION SETUP")
    print(f"{'='*70}")
    print(f"Model: {embedding_config.model_name}")
    print(f"Device: {embedding_config.device}")
    print(f"Batch size: {embedding_config.batch_size}")
    print(f"Normalize: {embedding_config.normalize}")
    print(f"{'='*70}\n")

    # Initialize embedder
    embedder = SentenceTransformerEmbedder(
        model_name=embedding_config.model_name,
        device=embedding_config.device,
        cache_dir=embedding_config.cache_dir
    )

    # Initialize generator
    generator = JobEmbeddingGenerator(embedder=embedder)

    # Generate embeddings
    result = generator.generate_embeddings(
        batch_size=embedding_config.batch_size,
        show_progress=True
    )

    print(f"\n✓ Embeddings generated successfully!")
    print(f"  Model: {result['model_name']}")
    print(f"  Jobs: {result['num_jobs']}")
    print(f"  Total embeddings: {result['total_embeddings']}")
    print(f"  Output: {result['output_path']}")
    print(f"\nNext steps:")
    print(f"  - Use embeddings for semantic job search")
    print(f"  - Run CV matching against job embeddings")
    print(f"  - Test search quality with diverse queries\n")

    return result


if __name__ == "__main__":
    main()
