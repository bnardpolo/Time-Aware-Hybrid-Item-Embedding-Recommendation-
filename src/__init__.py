"""
Time-Aware Hybrid Item Embedding Recommendation System

A production-ready hybrid recommender combining behavioral (SPPMI) and
semantic (SBERT) signals with time-based recency weighting.
"""

__version__ = "1.0.0"

# Import main classes and functions
from .preprocessing import (
    load_and_preprocess_data,
    create_baskets,
    extract_product_text,
    create_time_split
)

from .embeddings import (
    build_sppmi_embeddings,
    build_text_embeddings,
    combine_embeddings,
    build_hybrid_embeddings
)

from .models import (
    build_customer_vectors,
    build_faiss_index,
    HybridRecommender
)

from .evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommendations,
    evaluate_model,
    grid_search_weights,
    find_best_weights
)

from .utils import (
    plot_metric_heatmaps,
    plot_model_comparison,
    export_embeddings,
    validate_embeddings
)

__all__ = [
    # Preprocessing
    'load_and_preprocess_data',
    'create_baskets',
    'extract_product_text',
    'create_time_split',

    # Embeddings
    'build_sppmi_embeddings',
    'build_text_embeddings',
    'combine_embeddings',
    'build_hybrid_embeddings',

    # Models
    'build_customer_vectors',
    'build_faiss_index',
    'HybridRecommender',

    # Evaluation
    'precision_at_k',
    'recall_at_k',
    'ndcg_at_k',
    'evaluate_recommendations',
    'evaluate_model',
    'grid_search_weights',
    'find_best_weights',

    # Utils
    'plot_metric_heatmaps',
    'plot_model_comparison',
    'export_embeddings',
    'validate_embeddings',
]
