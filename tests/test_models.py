"""Tests for models module."""

import pytest
import numpy as np
import pandas as pd
from src.models import (
    build_customer_vectors,
    build_faiss_index,
    HybridRecommender
)


def test_build_customer_vectors(sample_transactions, sample_embeddings):
    """Test customer vector building."""
    result = build_customer_vectors(sample_transactions, sample_embeddings, halflife_days=30)

    assert isinstance(result, pd.DataFrame)
    assert 'customerid' in result.columns
    assert 'embedding' in result.columns
    assert len(result) > 0

    # Check normalization
    norms = result['embedding'].apply(lambda x: np.linalg.norm(x))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_faiss_index(sample_embeddings):
    """Test FAISS index building."""
    index, item_ids = build_faiss_index(sample_embeddings)

    assert index.ntotal == len(sample_embeddings)
    assert len(item_ids) == len(sample_embeddings)
    assert item_ids == sample_embeddings['stockcode'].astype(str).tolist()


def test_hybrid_recommender_init(sample_embeddings):
    """Test HybridRecommender initialization."""
    recommender = HybridRecommender(sample_embeddings)

    assert recommender.item_embeddings is not None
    assert recommender.faiss_index is not None
    assert recommender.item_id_list is not None
    assert recommender.customer_vectors is None


def test_hybrid_recommender_fit(sample_transactions, sample_embeddings):
    """Test HybridRecommender fitting."""
    recommender = HybridRecommender(sample_embeddings)
    recommender.fit(sample_transactions, halflife_days=30)

    assert recommender.customer_vectors is not None
    assert len(recommender.customer_vectors) > 0


def test_hybrid_recommender_recommend(sample_transactions, sample_embeddings):
    """Test recommendation generation."""
    recommender = HybridRecommender(sample_embeddings)
    recommender.fit(sample_transactions, halflife_days=30)

    customer_id = sample_transactions['customerid'].iloc[0]
    recommendations = recommender.recommend(customer_id, n_recommendations=3)

    assert isinstance(recommendations, list)
    assert len(recommendations) <= 3
    assert all(isinstance(item, str) for item in recommendations)


def test_hybrid_recommender_recommend_not_fitted(sample_embeddings):
    """Test that recommend raises error if not fitted."""
    recommender = HybridRecommender(sample_embeddings)

    with pytest.raises(ValueError, match="not fitted"):
        recommender.recommend(customer_id=1001, n_recommendations=10)


def test_hybrid_recommender_recommend_unknown_customer(sample_transactions, sample_embeddings):
    """Test recommendation for unknown customer."""
    recommender = HybridRecommender(sample_embeddings)
    recommender.fit(sample_transactions, halflife_days=30)

    with pytest.raises(ValueError, match="not found"):
        recommender.recommend(customer_id=99999, n_recommendations=10)


def test_hybrid_recommender_get_similar_items(sample_embeddings):
    """Test similar item search."""
    recommender = HybridRecommender(sample_embeddings)

    item_id = sample_embeddings['stockcode'].iloc[0]
    similar_items = recommender.get_similar_items(item_id, n_similar=3)

    assert isinstance(similar_items, list)
    assert len(similar_items) <= 3
    assert item_id not in similar_items  # Should exclude the query item


def test_hybrid_recommender_save_load(sample_transactions, sample_embeddings, tmp_path):
    """Test model save and load."""
    # Train model
    recommender = HybridRecommender(sample_embeddings)
    recommender.fit(sample_transactions, halflife_days=30)

    # Save model
    save_path = tmp_path / "model.pkl"
    recommender.save(save_path)

    assert save_path.exists()

    # Load model
    loaded_recommender = HybridRecommender.load(save_path)

    assert loaded_recommender.item_embeddings is not None
    assert loaded_recommender.customer_vectors is not None
    assert loaded_recommender.faiss_index is not None

    # Test that loaded model works
    customer_id = sample_transactions['customerid'].iloc[0]
    recommendations = loaded_recommender.recommend(customer_id, n_recommendations=3)
    assert isinstance(recommendations, list)


def test_hybrid_recommender_batch_recommend(sample_transactions, sample_embeddings):
    """Test batch recommendation."""
    recommender = HybridRecommender(sample_embeddings)
    recommender.fit(sample_transactions, halflife_days=30)

    customer_ids = sample_transactions['customerid'].unique().tolist()
    results = recommender.recommend_batch(customer_ids, n_recommendations=3)

    assert isinstance(results, dict)
    assert len(results) == len(customer_ids)
    assert all(isinstance(recs, list) for recs in results.values())
