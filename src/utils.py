"""
Utility functions for the recommendation system.

This module provides helper functions for visualization, data export,
and other common operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def setup_plotting_style():
    """Set up consistent plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['font.size'] = 10


def plot_metric_heatmaps(results_df, metrics=['precision@10', 'recall@10', 'ndcg@10'],
                         figsize=(20, 5), save_path=None):
    """
    Create heatmaps for metrics across weight combinations.

    Parameters
    ----------
    results_df : pd.DataFrame
        Grid search results with alpha, beta, and metric columns
    metrics : list, default=['precision@10', 'recall@10', 'ndcg@10']
        Metrics to plot
    figsize : tuple, default=(20, 5)
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    for idx, metric in enumerate(metrics):
        # Create pivot table
        pivot = results_df.pivot_table(
            index='beta',
            columns='alpha',
            values=metric,
            aggfunc='mean'
        )

        # Create heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap='YlOrRd',
            ax=axes[idx],
            cbar_kws={'label': metric.upper()}
        )

        axes[idx].set_title(f'{metric.upper()} by Weight Combination',
                           fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Alpha (Text Weight)', fontsize=12)
        axes[idx].set_ylabel('Beta (Basket Weight)', fontsize=12)
        axes[idx].invert_yaxis()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmaps saved to: {save_path}")

    return fig


def plot_model_comparison(comparison_df, metrics=['Precision@10', 'Recall@10', 'NDCG@10'],
                         figsize=(18, 5), save_path=None):
    """
    Create bar charts comparing different models.

    Parameters
    ----------
    comparison_df : pd.DataFrame
        Comparison dataframe with Model column and metric columns
    metrics : list
        Metrics to plot
    figsize : tuple, default=(18, 5)
        Figure size
    save_path : str or Path, optional
        Path to save figure

    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)
    if len(metrics) == 1:
        axes = [axes]

    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']

    for idx, metric in enumerate(metrics):
        values = comparison_df[metric].values
        bars = axes[idx].bar(range(len(comparison_df)), values,
                           color=colors[:len(comparison_df)],
                           alpha=0.7, edgecolor='black', linewidth=1.5)

        axes[idx].set_xticks(range(len(comparison_df)))
        axes[idx].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
        axes[idx].set_ylabel(metric, fontsize=12)
        axes[idx].set_title(f'{metric} Comparison', fontsize=14, fontweight='bold')
        axes[idx].grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            axes[idx].text(bar.get_x() + bar.get_width()/2., height,
                         f'{height:.4f}',
                         ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Highlight the best model
        axes[idx].patches[-1].set_edgecolor('gold')
        axes[idx].patches[-1].set_linewidth(3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")

    return fig


def export_embeddings(item_embeddings_df, output_path, format='csv'):
    """
    Export item embeddings to file.

    Parameters
    ----------
    item_embeddings_df : pd.DataFrame
        DataFrame with stockcode and embedding columns
    output_path : str or Path
        Output file path
    format : str, default='csv'
        Output format ('csv', 'parquet', 'pickle')
    """
    output_path = Path(output_path)

    if format == 'csv':
        # Expand embeddings into columns
        embeddings_expanded = pd.DataFrame(
            item_embeddings_df['embedding'].tolist(),
            columns=[f'emb_{i}' for i in range(len(item_embeddings_df['embedding'].iloc[0]))]
        )
        embeddings_expanded.insert(0, 'stockcode', item_embeddings_df['stockcode'])
        embeddings_expanded.to_csv(output_path, index=False)

    elif format == 'parquet':
        item_embeddings_df.to_parquet(output_path, index=False)

    elif format == 'pickle':
        item_embeddings_df.to_pickle(output_path)

    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Embeddings exported to: {output_path}")


def create_recommendation_report(recommender, customer_id, product_text_df,
                                n_recommendations=10):
    """
    Create a detailed recommendation report for a customer.

    Parameters
    ----------
    recommender : HybridRecommender
        Trained recommender model
    customer_id : int
        Customer ID
    product_text_df : pd.DataFrame
        Product text with stockcode and description columns
    n_recommendations : int, default=10
        Number of recommendations

    Returns
    -------
    pd.DataFrame
        Detailed recommendation report
    """
    # Get recommendations
    recommended_items = recommender.recommend(customer_id, n_recommendations)

    # Add product descriptions
    report = []
    for rank, item_id in enumerate(recommended_items, 1):
        product_info = product_text_df[product_text_df['stockcode'] == item_id]
        description = product_info['description'].values[0] if len(product_info) > 0 else 'N/A'

        report.append({
            'rank': rank,
            'stockcode': item_id,
            'description': description
        })

    return pd.DataFrame(report)


def print_summary_statistics(df, title="Dataset Summary"):
    """
    Print formatted summary statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to summarize
    title : str
        Title for the summary
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    print(f"\nColumn types:")
    print(df.dtypes.value_counts())
    print("\n" + "=" * 80)


def validate_embeddings(embeddings_df, embedding_col='embedding', expected_dim=None):
    """
    Validate embedding dataframe.

    Parameters
    ----------
    embeddings_df : pd.DataFrame
        Dataframe with embeddings
    embedding_col : str, default='embedding'
        Name of embedding column
    expected_dim : int, optional
        Expected embedding dimension

    Returns
    -------
    dict
        Validation results
    """
    results = {
        'valid': True,
        'errors': []
    }

    # Check if column exists
    if embedding_col not in embeddings_df.columns:
        results['valid'] = False
        results['errors'].append(f"Column '{embedding_col}' not found")
        return results

    # Check for null values
    if embeddings_df[embedding_col].isna().any():
        results['valid'] = False
        results['errors'].append("Null values found in embeddings")

    # Check embedding dimensions
    dims = embeddings_df[embedding_col].apply(lambda x: len(x) if hasattr(x, '__len__') else 0)
    if dims.nunique() > 1:
        results['valid'] = False
        results['errors'].append(f"Inconsistent dimensions: {dims.unique()}")

    if expected_dim and (dims != expected_dim).any():
        results['valid'] = False
        results['errors'].append(f"Expected dimension {expected_dim}, found {dims.unique()}")

    # Check for proper normalization (L2 norm ≈ 1)
    norms = embeddings_df[embedding_col].apply(lambda x: np.linalg.norm(x))
    if not np.allclose(norms, 1.0, atol=1e-4):
        results['valid'] = False
        results['errors'].append(f"Embeddings not normalized (norms: {norms.min():.4f} - {norms.max():.4f})")

    return results


def benchmark_inference_speed(recommender, customer_ids, n_runs=100):
    """
    Benchmark recommendation inference speed.

    Parameters
    ----------
    recommender : HybridRecommender
        Trained recommender
    customer_ids : list
        List of customer IDs to test
    n_runs : int, default=100
        Number of runs for averaging

    Returns
    -------
    dict
        Benchmark results
    """
    import time

    times = []

    for _ in range(n_runs):
        customer_id = np.random.choice(customer_ids)
        start = time.time()
        try:
            _ = recommender.recommend(customer_id, n_recommendations=10)
            elapsed = (time.time() - start) * 1000  # Convert to ms
            times.append(elapsed)
        except ValueError:
            continue

    if len(times) == 0:
        return {'error': 'No successful recommendations'}

    return {
        'mean_ms': np.mean(times),
        'median_ms': np.median(times),
        'p95_ms': np.percentile(times, 95),
        'p99_ms': np.percentile(times, 99),
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'n_runs': len(times)
    }
