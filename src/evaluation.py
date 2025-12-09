"""
Evaluation module for recommendation system.

This module provides metrics (Precision@K, Recall@K, NDCG@K) and
grid search functionality for hyperparameter optimization.
"""

import numpy as np
import pandas as pd
from typing import List, Set, Dict
from itertools import product
from tqdm import tqdm


def precision_at_k(recommended, actual, k):
    """
    Calculate Precision@K.

    Precision measures the fraction of recommended items that are relevant.

    Parameters
    ----------
    recommended : list
        List of recommended item IDs
    actual : set
        Set of actual relevant item IDs
    k : int
        Number of recommendations to consider

    Returns
    -------
    float
        Precision@K score
    """
    recommended_k = set(recommended[:k])
    return len(recommended_k & actual) / k if k > 0 else 0.0


def recall_at_k(recommended, actual, k):
    """
    Calculate Recall@K.

    Recall measures the fraction of relevant items that were recommended.

    Parameters
    ----------
    recommended : list
        List of recommended item IDs
    actual : set
        Set of actual relevant item IDs
    k : int
        Number of recommendations to consider

    Returns
    -------
    float
        Recall@K score
    """
    recommended_k = set(recommended[:k])
    return len(recommended_k & actual) / len(actual) if len(actual) > 0 else 0.0


def ndcg_at_k(recommended, actual, k):
    """
    Calculate NDCG@K (Normalized Discounted Cumulative Gain).

    NDCG accounts for the ranking position, giving more weight to
    correct recommendations at the top of the list.

    Parameters
    ----------
    recommended : list
        List of recommended item IDs
    actual : set
        Set of actual relevant item IDs
    k : int
        Number of recommendations to consider

    Returns
    -------
    float
        NDCG@K score
    """
    recommended_k = recommended[:k]

    # Calculate DCG
    dcg = 0.0
    for i, item in enumerate(recommended_k, 1):
        if item in actual:
            dcg += 1.0 / np.log2(i + 1)

    # Calculate IDCG (ideal DCG)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))

    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommendations(customer_id, recommended_items, ground_truth, k_values=[5, 10, 20]):
    """
    Evaluate recommendations for a single customer across multiple K values.

    Parameters
    ----------
    customer_id : int
        Customer ID
    recommended_items : list
        List of recommended item IDs
    ground_truth : dict
        Dictionary mapping customer_id to set of actual items
    k_values : list, default=[5, 10, 20]
        List of K values to evaluate

    Returns
    -------
    dict
        Dictionary with metrics for each K
    """
    actual = ground_truth.get(customer_id, set())

    results = {}
    for k in k_values:
        results[f'precision@{k}'] = precision_at_k(recommended_items, actual, k)
        results[f'recall@{k}'] = recall_at_k(recommended_items, actual, k)
        results[f'ndcg@{k}'] = ndcg_at_k(recommended_items, actual, k)

    return results


def evaluate_model(recommender, ground_truth, k_values=[5, 10, 20], customer_subset=None):
    """
    Evaluate a recommender model on a test set.

    Parameters
    ----------
    recommender : HybridRecommender
        Trained recommender model
    ground_truth : dict
        Dictionary mapping customer_id to set of actual items
    k_values : list, default=[5, 10, 20]
        List of K values to evaluate
    customer_subset : list, optional
        Subset of customers to evaluate (for speed)

    Returns
    -------
    pd.DataFrame
        DataFrame with average metrics
    """
    test_customers = customer_subset if customer_subset is not None else list(ground_truth.keys())

    all_results = []

    for customer_id in tqdm(test_customers, desc="Evaluating"):
        try:
            recommended_items = recommender.recommend(customer_id, n_recommendations=max(k_values))
            metrics = evaluate_recommendations(customer_id, recommended_items, ground_truth, k_values)
            all_results.append(metrics)
        except ValueError:
            # Customer not found
            continue

    if len(all_results) == 0:
        raise ValueError("No customers could be evaluated")

    # Calculate average metrics
    avg_metrics = {}
    for key in all_results[0].keys():
        avg_metrics[key] = np.mean([r[key] for r in all_results])

    avg_metrics['n_customers'] = len(all_results)

    return pd.DataFrame([avg_metrics])


def grid_search_weights(basket_embeddings, text_embeddings, transactions_df, ground_truth,
                       alpha_values=None, beta_values=None, embedding_dim=128,
                       halflife_days=60, k_values=[5, 10, 20], sample_customers=None):
    """
    Perform grid search over alpha and beta weight combinations.

    Parameters
    ----------
    basket_embeddings : pd.DataFrame
        Basket embeddings (SPPMI)
    text_embeddings : pd.DataFrame
        Text embeddings (SBERT)
    transactions_df : pd.DataFrame
        Transaction data for building customer vectors
    ground_truth : dict
        Test set ground truth
    alpha_values : list, optional
        Alpha values to test (default: 0.0 to 1.0 in steps of 0.1)
    beta_values : list, optional
        Beta values to test (default: 0.0 to 1.0 in steps of 0.1)
    embedding_dim : int, default=128
        Final embedding dimension
    halflife_days : int, default=60
        Recency decay half-life
    k_values : list, default=[5, 10, 20]
        K values for evaluation
    sample_customers : list, optional
        Subset of customers to evaluate (for speed)

    Returns
    -------
    pd.DataFrame
        DataFrame with results for all weight combinations
    """
    from .embeddings import combine_embeddings
    from .models import HybridRecommender

    # Default ranges
    if alpha_values is None:
        alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    if beta_values is None:
        beta_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print(f"Testing {len(alpha_values)} x {len(beta_values)} = {len(alpha_values) * len(beta_values)} combinations")
    if sample_customers:
        print(f"Evaluating on {len(sample_customers)} customers")

    results = []

    for alpha, beta in tqdm(list(product(alpha_values, beta_values)), desc="Grid search"):
        # Create weighted embeddings
        item_embeddings = combine_embeddings(
            basket_embeddings,
            text_embeddings,
            alpha=alpha,
            beta=beta,
            final_dim=embedding_dim
        )

        # Build recommender
        recommender = HybridRecommender(item_embeddings)
        recommender.fit(transactions_df, halflife_days=halflife_days)

        # Evaluate
        try:
            eval_results = evaluate_model(
                recommender,
                ground_truth,
                k_values=k_values,
                customer_subset=sample_customers
            )

            # Add weights to results
            eval_results['alpha'] = alpha
            eval_results['beta'] = beta

            results.append(eval_results)

        except Exception as e:
            print(f"Error evaluating alpha={alpha}, beta={beta}: {e}")
            continue

    if len(results) == 0:
        raise ValueError("Grid search failed - no valid results")

    # Concatenate all results
    results_df = pd.concat(results, ignore_index=True)

    return results_df


def find_best_weights(results_df, metric='ndcg@10'):
    """
    Find the best weight combination from grid search results.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from grid_search_weights
    metric : str, default='ndcg@10'
        Metric to optimize

    Returns
    -------
    dict
        Best weights and their performance
    """
    best_idx = results_df[metric].idxmax()
    best_row = results_df.loc[best_idx]

    return {
        'alpha': best_row['alpha'],
        'beta': best_row['beta'],
        'metrics': {
            col: best_row[col]
            for col in results_df.columns
            if '@' in col
        }
    }


def compare_models(results_df, alpha_beta_pairs, metric='ndcg@10'):
    """
    Compare specific model configurations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Results from grid search
    alpha_beta_pairs : list of tuples
        List of (alpha, beta) pairs to compare
    metric : str, default='ndcg@10'
        Primary metric for comparison

    Returns
    -------
    pd.DataFrame
        Comparison table
    """
    comparison = []

    for alpha, beta in alpha_beta_pairs:
        row = results_df[(results_df['alpha'] == alpha) & (results_df['beta'] == beta)]
        if len(row) > 0:
            row = row.iloc[0]
            comparison.append({
                'alpha': alpha,
                'beta': beta,
                **{col: row[col] for col in results_df.columns if '@' in col}
            })

    return pd.DataFrame(comparison)


def calculate_improvement(results_df, baseline_alpha, baseline_beta, best_alpha, best_beta, metric='ndcg@10'):
    """
    Calculate improvement over baseline.

    Parameters
    ----------
    results_df : pd.DataFrame
        Grid search results
    baseline_alpha : float
        Baseline alpha value
    baseline_beta : float
        Baseline beta value
    best_alpha : float
        Best alpha value
    best_beta : float
        Best beta value
    metric : str, default='ndcg@10'
        Metric to compare

    Returns
    -------
    float
        Percentage improvement
    """
    baseline_row = results_df[(results_df['alpha'] == baseline_alpha) &
                              (results_df['beta'] == baseline_beta)]
    best_row = results_df[(results_df['alpha'] == best_alpha) &
                          (results_df['beta'] == best_beta)]

    if len(baseline_row) == 0 or len(best_row) == 0:
        raise ValueError("Baseline or best configuration not found")

    baseline_score = baseline_row.iloc[0][metric]
    best_score = best_row.iloc[0][metric]

    if baseline_score == 0:
        return float('inf')

    return ((best_score - baseline_score) / baseline_score) * 100
