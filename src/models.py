"""
Recommendation model module.

This module implements customer vector aggregation, FAISS indexing,
and the main recommendation engine.
"""

import numpy as np
import pandas as pd
import faiss
from typing import List, Dict, Tuple, Optional


def build_customer_vectors(transactions_df, item_embeddings_df, halflife_days=60):
    """
    Aggregate item embeddings to customer level with recency weighting.

    Recent purchases are weighted more heavily using exponential decay.

    Parameters
    ----------
    transactions_df : pd.DataFrame
        Clean transaction data with columns: customerid, stockcode, revenue, invoicedate
    item_embeddings_df : pd.DataFrame
        Item embeddings with stockcode and embedding columns
    halflife_days : int, default=60
        Half-life for exponential decay (in days)

    Returns
    -------
    pd.DataFrame
        DataFrame with customerid and embedding columns
    """
    df = transactions_df.copy()

    # Calculate recency decay
    max_date = df['invoicedate'].max()
    lambda_decay = np.log(2) / halflife_days
    df['days_ago'] = (max_date - df['invoicedate']).dt.days
    df['recency_weight'] = np.exp(-lambda_decay * df['days_ago'])

    # Cap revenue for weighting
    revenue_cap = df['revenue'].quantile(0.99)
    df['revenue_capped'] = df['revenue'].clip(upper=revenue_cap)

    # Combined weight
    df['weight'] = df['revenue_capped'] * df['recency_weight']

    # Map embeddings to transactions
    embedding_map = dict(zip(
        item_embeddings_df['stockcode'],
        item_embeddings_df['embedding']
    ))

    df['item_embedding'] = df['stockcode'].astype(str).map(embedding_map)

    # Drop transactions without embeddings
    df = df.dropna(subset=['item_embedding'])

    print(f"Transactions with embeddings: {len(df):,}")

    # Aggregate to customer level
    customer_vectors = []

    for customer_id, group in df.groupby('customerid'):
        # Stack embeddings and weights
        embeddings_matrix = np.vstack(group['item_embedding'].values)
        weights = group['weight'].values.reshape(-1, 1)

        # Weighted average
        weighted_sum = (embeddings_matrix * weights).sum(axis=0)
        total_weight = weights.sum()
        customer_vector = weighted_sum / (total_weight + 1e-9)

        # Normalize
        customer_vector = customer_vector / (np.linalg.norm(customer_vector) + 1e-9)

        customer_vectors.append({
            'customerid': int(customer_id),
            'embedding': customer_vector
        })

    result = pd.DataFrame(customer_vectors)

    print(f"Customers with vectors: {len(result):,}")

    return result


def build_faiss_index(item_embeddings_df):
    """
    Build FAISS index for efficient nearest neighbor search.

    Uses inner product for cosine similarity on normalized vectors.

    Parameters
    ----------
    item_embeddings_df : pd.DataFrame
        DataFrame with stockcode and embedding columns

    Returns
    -------
    tuple
        (faiss_index, item_id_list) where item_id_list maps indices to stockcodes
    """
    # Extract embeddings and IDs
    item_ids = item_embeddings_df['stockcode'].astype(str).tolist()
    embeddings_matrix = np.vstack(item_embeddings_df['embedding'].values).astype('float32')

    # Verify normalization
    norms = np.linalg.norm(embeddings_matrix, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), "Embeddings must be L2 normalized"

    # Create FAISS index (Inner Product = Cosine Similarity for normalized vectors)
    dimension = embeddings_matrix.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_matrix)

    print(f"FAISS index built:")
    print(f"  Dimension: {dimension}")
    print(f"  Total items: {index.ntotal:,}")

    return index, item_ids


class HybridRecommender:
    """
    Hybrid recommendation engine combining behavioral and semantic signals.

    This class encapsulates the full recommendation pipeline including
    customer vector creation, FAISS indexing, and recommendation generation.

    Attributes
    ----------
    item_embeddings : pd.DataFrame
        Item embeddings
    customer_vectors : pd.DataFrame
        Customer preference vectors
    faiss_index : faiss.Index
        FAISS index for fast similarity search
    item_id_list : list
        Mapping from FAISS indices to item IDs
    """

    def __init__(self, item_embeddings_df):
        """
        Initialize recommender with item embeddings.

        Parameters
        ----------
        item_embeddings_df : pd.DataFrame
            DataFrame with stockcode and embedding columns
        """
        self.item_embeddings = item_embeddings_df
        self.customer_vectors = None
        self.faiss_index = None
        self.item_id_list = None

        # Build FAISS index
        self.faiss_index, self.item_id_list = build_faiss_index(item_embeddings_df)

    def fit(self, transactions_df, halflife_days=60):
        """
        Build customer vectors from transaction history.

        Parameters
        ----------
        transactions_df : pd.DataFrame
            Transaction data
        halflife_days : int, default=60
            Recency decay half-life in days

        Returns
        -------
        self
        """
        self.customer_vectors = build_customer_vectors(
            transactions_df,
            self.item_embeddings,
            halflife_days=halflife_days
        )
        return self

    def recommend(self, customer_id, n_recommendations=10, exclude_items=None):
        """
        Generate recommendations for a specific customer.

        Parameters
        ----------
        customer_id : int
            Customer ID to generate recommendations for
        n_recommendations : int, default=10
            Number of items to recommend
        exclude_items : set or list, optional
            Items to exclude from recommendations (e.g., already purchased)

        Returns
        -------
        list
            List of recommended item IDs
        """
        if self.customer_vectors is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get customer embedding
        cust_row = self.customer_vectors[self.customer_vectors['customerid'] == customer_id]
        if len(cust_row) == 0:
            raise ValueError(f"Customer {customer_id} not found")

        query_vec = cust_row['embedding'].values[0].reshape(1, -1).astype('float32')

        # Find nearest items (search for more if we need to exclude some)
        search_k = n_recommendations * 3 if exclude_items else n_recommendations
        distances, indices = self.faiss_index.search(query_vec, search_k)

        # Map to item IDs
        recommended_items = [self.item_id_list[idx] for idx in indices[0]]

        # Exclude items if specified
        if exclude_items:
            exclude_set = set(exclude_items) if not isinstance(exclude_items, set) else exclude_items
            recommended_items = [item for item in recommended_items if item not in exclude_set]

        return recommended_items[:n_recommendations]

    def recommend_batch(self, customer_ids, n_recommendations=10):
        """
        Generate recommendations for multiple customers.

        Parameters
        ----------
        customer_ids : list
            List of customer IDs
        n_recommendations : int, default=10
            Number of items to recommend per customer

        Returns
        -------
        dict
            Dictionary mapping customer_id to list of recommendations
        """
        results = {}

        for customer_id in customer_ids:
            try:
                results[customer_id] = self.recommend(customer_id, n_recommendations)
            except ValueError:
                # Customer not found
                results[customer_id] = []

        return results

    def get_similar_items(self, item_id, n_similar=10):
        """
        Find items similar to a given item.

        Parameters
        ----------
        item_id : str
            Item ID to find similar items for
        n_similar : int, default=10
            Number of similar items to return

        Returns
        -------
        list
            List of similar item IDs (excluding the query item itself)
        """
        # Get item embedding
        item_row = self.item_embeddings[self.item_embeddings['stockcode'] == str(item_id)]
        if len(item_row) == 0:
            raise ValueError(f"Item {item_id} not found")

        item_vec = item_row['embedding'].values[0].reshape(1, -1).astype('float32')

        # Search for similar items (add 1 to exclude the item itself)
        distances, indices = self.faiss_index.search(item_vec, n_similar + 1)

        # Map to item IDs and exclude the query item
        similar_items = [
            self.item_id_list[idx] for idx in indices[0]
            if self.item_id_list[idx] != str(item_id)
        ]

        return similar_items[:n_similar]

    def save(self, filepath):
        """
        Save the recommender model to disk.

        Parameters
        ----------
        filepath : str or Path
            Path to save the model
        """
        import pickle

        model_data = {
            'item_embeddings': self.item_embeddings,
            'customer_vectors': self.customer_vectors,
            'faiss_index': faiss.serialize_index(self.faiss_index),
            'item_id_list': self.item_id_list
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Load a recommender model from disk.

        Parameters
        ----------
        filepath : str or Path
            Path to load the model from

        Returns
        -------
        HybridRecommender
            Loaded recommender instance
        """
        import pickle

        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        # Create instance
        recommender = cls(model_data['item_embeddings'])
        recommender.customer_vectors = model_data['customer_vectors']
        recommender.faiss_index = faiss.deserialize_index(model_data['faiss_index'])
        recommender.item_id_list = model_data['item_id_list']

        print(f"Model loaded from {filepath}")

        return recommender
