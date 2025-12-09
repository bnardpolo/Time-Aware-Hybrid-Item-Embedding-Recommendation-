"""Tests for evaluation module."""

import pytest
import numpy as np
from src.evaluation import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    evaluate_recommendations,
    find_best_weights,
    calculate_improvement
)


def test_precision_at_k():
    """Test Precision@K metric."""
    recommended = ['A', 'B', 'C', 'D', 'E']
    actual = {'A', 'C', 'F'}

    # 2 out of 5 are relevant
    assert precision_at_k(recommended, actual, k=5) == pytest.approx(0.4)

    # 2 out of 3 are relevant
    assert precision_at_k(recommended, actual, k=3) == pytest.approx(2/3)

    # 0 out of 2 are relevant
    assert precision_at_k(['D', 'E'], actual, k=2) == 0.0


def test_recall_at_k():
    """Test Recall@K metric."""
    recommended = ['A', 'B', 'C', 'D', 'E']
    actual = {'A', 'C', 'F'}

    # 2 out of 3 relevant items found
    assert recall_at_k(recommended, actual, k=5) == pytest.approx(2/3)

    # 2 out of 3 relevant items found
    assert recall_at_k(recommended, actual, k=3) == pytest.approx(2/3)

    # 1 out of 3 relevant items found
    assert recall_at_k(recommended, actual, k=1) == pytest.approx(1/3)


def test_ndcg_at_k():
    """Test NDCG@K metric."""
    # Perfect ranking
    recommended = ['A', 'B', 'C']
    actual = {'A', 'B', 'C'}
    assert ndcg_at_k(recommended, actual, k=3) == pytest.approx(1.0)

    # Worst ranking (all relevant items at the end)
    recommended = ['D', 'E', 'F', 'A', 'B']
    actual = {'A', 'B'}
    assert ndcg_at_k(recommended, actual, k=5) < ndcg_at_k(['A', 'B', 'C', 'D', 'E'], actual, k=5)

    # No relevant items
    recommended = ['D', 'E', 'F']
    actual = {'A', 'B', 'C'}
    assert ndcg_at_k(recommended, actual, k=3) == 0.0


def test_evaluate_recommendations(sample_ground_truth):
    """Test evaluation of recommendations."""
    customer_id = 1001
    recommended = ['A001', 'A002', 'A003', 'A004', 'A005']

    results = evaluate_recommendations(customer_id, recommended, sample_ground_truth, k_values=[3, 5])

    assert 'precision@3' in results
    assert 'recall@3' in results
    assert 'ndcg@3' in results
    assert 'precision@5' in results

    # All recommended items match actual
    assert results['precision@3'] == pytest.approx(1.0)
    assert results['recall@3'] == pytest.approx(1.0)


def test_find_best_weights():
    """Test finding best weights from grid search results."""
    results_df = pd.DataFrame({
        'alpha': [0.0, 0.5, 1.0],
        'beta': [0.0, 0.5, 1.0],
        'precision@10': [0.3, 0.4, 0.35],
        'ndcg@10': [0.4, 0.5, 0.45]
    })

    best = find_best_weights(results_df, metric='ndcg@10')

    assert best['alpha'] == 0.5
    assert best['beta'] == 0.5
    assert 'metrics' in best
    assert best['metrics']['ndcg@10'] == 0.5


def test_calculate_improvement():
    """Test improvement calculation."""
    import pandas as pd

    results_df = pd.DataFrame({
        'alpha': [0.0, 0.5, 1.0],
        'beta': [0.0, 0.5, 1.0],
        'ndcg@10': [0.4, 0.5, 0.45]
    })

    improvement = calculate_improvement(
        results_df,
        baseline_alpha=0.0,
        baseline_beta=0.0,
        best_alpha=0.5,
        best_beta=0.5,
        metric='ndcg@10'
    )

    assert improvement == pytest.approx(25.0)  # (0.5 - 0.4) / 0.4 * 100


def test_precision_recall_edge_cases():
    """Test edge cases for metrics."""
    # Empty actual set
    assert recall_at_k(['A', 'B'], set(), k=2) == 0.0

    # Empty recommended list
    assert precision_at_k([], {'A', 'B'}, k=0) == 0.0

    # No overlap
    assert precision_at_k(['A', 'B'], {'C', 'D'}, k=2) == 0.0
    assert recall_at_k(['A', 'B'], {'C', 'D'}, k=2) == 0.0


# Import pandas for test_find_best_weights
import pandas as pd
