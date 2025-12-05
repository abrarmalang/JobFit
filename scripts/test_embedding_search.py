"""
Test Embedding Search

Quick test script to verify embedding generation and search functionality.
Run with: python scripts/test_embedding_search.py
"""

import sys
from pathlib import Path

# Add src directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))


def test_search():
    """Test embedding search functionality."""

    print("\n" + "="*70)
    print("TESTING EMBEDDING SEARCH")
    print("="*70 + "\n")

    try:
        from workers.embeddings.search import EmbeddingSearch
        from workers.embeddings.embedder_sentencetransformer import SentenceTransformerEmbedder
        from shared.config import load_config

        # Load config
        config = load_config()
        model_name = config.embedding.model_name

        print(f"Initializing search with model: {model_name}")

        # Initialize search (will load embeddings)
        search = EmbeddingSearch(model_name=model_name)

        # Get statistics
        stats = search.get_statistics()
        print(f"\nEmbedding Statistics:")
        print(f"  Status: {stats['status']}")
        print(f"  Total jobs: {stats['total_jobs']}")
        print(f"  Sources: {stats['sources']}")
        print(f"  Countries: {stats['countries']}")
        print(f"  Embedding types: {len(stats['embedding_types'])}")

        # Initialize embedder for queries
        print(f"\nInitializing embedder for queries...")
        embedder = SentenceTransformerEmbedder(
            model_name=model_name,
            device=config.embedding.device
        )

        # Test queries
        test_queries = [
            "Senior Python developer with Django experience",
            "Data scientist with machine learning skills",
            "Frontend developer React TypeScript",
            "DevOps engineer Kubernetes AWS"
        ]

        print(f"\n{'─'*70}")
        print("TEST QUERIES")
        print(f"{'─'*70}\n")

        for i, query in enumerate(test_queries, 1):
            print(f"\n[{i}] Query: \"{query}\"")
            print(f"{'─'*70}")

            # Search using full description embeddings
            results = search.search_by_text(
                query_text=query,
                embedder=embedder,
                embedding_type="embedding_full",
                top_k=5,
                min_score=0.0
            )

            if not results:
                print("  No results found")
                continue

            print(f"  Top 5 matches:\n")
            for rank, job in enumerate(results, 1):
                print(f"  {rank}. {job['title']} at {job['company']}")
                print(f"     Match: {job['match_score']:.1%} | {job['location_display']} | {job['source']}")

        # Test multi-aspect search
        print(f"\n{'─'*70}")
        print("MULTI-ASPECT SEARCH TEST")
        print(f"{'─'*70}\n")

        query_text = "Senior Python Backend Developer"
        print(f"Query: \"{query_text}\"\n")

        # Encode with both aspects
        query_full = embedder.encode_single(query_text + " experienced developer with strong backend skills")
        query_title = embedder.encode_single(query_text)

        results = search.search_multi_aspect(
            query_embedding_full=query_full,
            query_embedding_title=query_title,
            weights={'full': 0.6, 'title': 0.4},
            top_k=5
        )

        print("Top 5 matches with breakdown:\n")
        for rank, job in enumerate(results, 1):
            print(f"{rank}. {job['title']} at {job['company']}")
            print(f"   Overall: {job['match_score']:.1%}")
            print(f"   - Full match: {job['match_breakdown']['full_match']:.1%}")
            print(f"   - Title match: {job['match_breakdown']['title_match']:.1%}")
            print()

        print("="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70 + "\n")

        return True

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to:")
        print("1. Collect job data: python scripts/run_collection_diverse.py")
        print("2. Process data: python scripts/run_processing.py")
        print("3. Generate embeddings: python scripts/run_generate_embeddings.py")
        print()
        return False

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_search()
    sys.exit(0 if success else 1)
