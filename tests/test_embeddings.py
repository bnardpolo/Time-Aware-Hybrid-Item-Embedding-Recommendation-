"""Tests for embeddings module."""

import pytest
import numpy as np
import pandas as pd
from src.embeddings import (
    build_sppmi_embeddings,
    build_text_embeddings,
    combine_embeddings
)


def test_build_sppmi_embeddings(sample_baskets):
    """Test SPPMI embedding generation."""
    result = build_sppmi_embeddings(sample_baskets, dim=16, k_shift=1)

    assert isinstance(result, pd.DataFrame)
    assert 'stockcode' in result.columns
    assert 'basket_embedding' in result.columns
    assert len(result) > 0

    # Check embedding dimensions
    first_embedding = result['basket_embedding'].iloc[0]
    assert len(first_embedding) == 16

    # Check normalization
    norms = result['basket_embedding'].apply(lambda x: np.linalg.norm(x))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_build_text_embeddings_tfidf(sample_product_text):
    """Test text embedding generation with TF-IDF fallback."""
    result = build_text_embeddings(sample_product_text, fallback_to_tfidf=True)

    assert isinstance(result, pd.DataFrame)
    assert 'stockcode' in result.columns
    assert 'text_embedding' in result.columns
    assert len(result) == len(sample_product_text)

    # Check embedding dimensions
    first_embedding = result['text_embedding'].iloc[0]
    assert len(first_embedding) > 0

    # Check normalization
    norms = result['text_embedding'].apply(lambda x: np.linalg.norm(x))
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_combine_embeddings():
    """Test embedding combination with PCA fusion."""
    # Create sample embeddings
    np.random.seed(42)

    basket_emb = pd.DataFrame({
        'stockcode': ['A001', 'A002', 'A003'],
        'basket_embedding': [
            np.random.randn(32) / np.sqrt(32),
            np.random.randn(32) / np.sqrt(32),
            np.random.randn(32) / np.sqrt(32)
        ]
    })

    text_emb = pd.DataFrame({
        'stockcode': ['A001', 'A002', 'A003'],
        'text_embedding': [
            np.random.randn(64) / np.sqrt(64),
            np.random.randn(64) / np.sqrt(64),
            np.random.randn(64) / np.sqrt(64)
        ]
    })

    result = combine_embeddings(basket_emb, text_emb, alpha=0.5, beta=0.5, final_dim=16)

    assert isinstance(result, pd.DataFrame)
    assert 'stockcode' in result.columns
    assert 'embedding' in result.columns
    assert len(result) == 3

    # Check final dimension
    first_embedding = result['embedding'].iloc[0]
    assert len(first_embedding) == 16

    # Check normalization
    norms = result['embedding'].apply(lambda x: np.linalg.norm(x))
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_combine_embeddings_different_weights():
    """Test that different weights produce different results."""
    np.random.seed(42)

    basket_emb = pd.DataFrame({
        'stockcode': ['A001'],
        'basket_embedding': [np.random.randn(32) / np.sqrt(32)]
    })

    text_emb = pd.DataFrame({
        'stockcode': ['A001'],
        'text_embedding': [np.random.randn(64) / np.sqrt(64)]
    })

    result1 = combine_embeddings(basket_emb, text_emb, alpha=1.0, beta=0.0, final_dim=16)
    result2 = combine_embeddings(basket_emb, text_emb, alpha=0.0, beta=1.0, final_dim=16)

    emb1 = result1['embedding'].iloc[0]
    emb2 = result2['embedding'].iloc[0]

    # Embeddings should be different with different weights
    assert not np.allclose(emb1, emb2)
