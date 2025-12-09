"""
Embedding generation module for the recommendation system.

This module implements SPPMI (behavioral embeddings), SBERT/TF-IDF (semantic embeddings),
and hybrid fusion techniques.
"""

import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from sklearn.decomposition import TruncatedSVD, PCA


def build_sppmi_embeddings(baskets, dim=128, k_shift=5):
    """
    Build item embeddings using Shifted Positive PMI with SVD.

    SPPMI learns embeddings based on item co-occurrence patterns in baskets.
    Items that frequently appear together get similar embeddings.

    Parameters
    ----------
    baskets : list of lists
        Each inner list contains item IDs from one transaction
    dim : int, default=128
        Dimensionality of final embeddings
    k_shift : float, default=5
        SPPMI shift parameter (controls sparsity)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: stockcode, basket_embedding
    """
    # Count item frequencies and co-occurrences
    item_counts = Counter()
    pair_counts = defaultdict(int)

    for basket in baskets:
        # Get unique items in basket
        unique_items = list(dict.fromkeys(map(str, basket)))

        # Update item counts
        for item in unique_items:
            item_counts[item] += 1

        # Update pair counts (symmetric)
        for i in range(len(unique_items)):
            for j in range(i + 1, len(unique_items)):
                a, b = unique_items[i], unique_items[j]
                pair_counts[(a, b)] += 1
                pair_counts[(b, a)] += 1

    # Build vocabulary
    items = list(item_counts.keys())
    item_to_idx = {item: idx for idx, item in enumerate(items)}
    vocab_size = len(items)

    print(f"Vocabulary size: {vocab_size:,}")

    # Calculate SPPMI values
    total_pairs = sum(item_counts.values())
    sppmi_matrix = defaultdict(dict)

    for (item_a, item_b), count_ab in pair_counts.items():
        # Calculate PMI
        p_ab = count_ab / total_pairs
        p_a = item_counts[item_a] / total_pairs
        p_b = item_counts[item_b] / total_pairs

        pmi = np.log2((p_ab / (p_a * p_b + 1e-12)) + 1e-12)

        # Apply shift and keep only positive values
        sppmi = max(pmi - np.log2(k_shift), 0.0)

        if sppmi > 0:
            idx_a = item_to_idx[item_a]
            idx_b = item_to_idx[item_b]
            sppmi_matrix[idx_a][idx_b] = sppmi

    # Convert to dense matrix for SVD
    matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    for i, cols in sppmi_matrix.items():
        for j, val in cols.items():
            matrix[i, j] = val

    print(f"Matrix sparsity: {(matrix == 0).mean()*100:.1f}%")

    # Apply SVD dimensionality reduction
    svd = TruncatedSVD(n_components=dim, random_state=42)
    embeddings = svd.fit_transform(matrix)

    # L2 normalize
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

    print(f"Explained variance: {svd.explained_variance_ratio_.sum()*100:.1f}%")

    # Create output dataframe
    result = pd.DataFrame({
        'stockcode': items,
        'basket_embedding': list(embeddings)
    })

    return result


def build_text_embeddings(product_text_df, model_name='all-MiniLM-L6-v2', fallback_to_tfidf=True):
    """
    Generate semantic embeddings from product descriptions using SBERT.

    Falls back to TF-IDF if SBERT is unavailable.

    Parameters
    ----------
    product_text_df : pd.DataFrame
        DataFrame with columns: stockcode, description
    model_name : str, default='all-MiniLM-L6-v2'
        Name of the Sentence-BERT model to use
    fallback_to_tfidf : bool, default=True
        Whether to fall back to TF-IDF if SBERT fails

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: stockcode, text_embedding
    """
    try:
        from sentence_transformers import SentenceTransformer

        print(f"Loading SBERT model: {model_name}")
        model = SentenceTransformer(model_name)

        # Encode descriptions
        descriptions = product_text_df['description'].fillna('no description').tolist()
        embeddings = model.encode(
            descriptions,
            batch_size=256,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        method = 'SBERT'

    except Exception as e:
        if not fallback_to_tfidf:
            raise

        print(f"SBERT unavailable: {e}")
        print("Falling back to TF-IDF")

        from sklearn.feature_extraction.text import TfidfVectorizer

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer(
            min_df=2,
            max_df=0.9,
            ngram_range=(1, 2),
            max_features=5000
        )

        descriptions = product_text_df['description'].fillna('no description').tolist()
        tfidf_matrix = vectorizer.fit_transform(descriptions)

        # Reduce dimensionality
        svd = TruncatedSVD(n_components=256, random_state=42)
        embeddings = svd.fit_transform(tfidf_matrix)

        # Normalize
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9)

        method = 'TF-IDF'

    # Create output dataframe
    result = pd.DataFrame({
        'stockcode': product_text_df['stockcode'].astype(str),
        'text_embedding': list(embeddings)
    })

    print(f"Text embeddings created using {method}")
    print(f"Embedding dimension: {embeddings.shape[1]}")

    return result


def combine_embeddings(basket_emb_df, text_emb_df, alpha=1.0, beta=1.0, final_dim=128):
    """
    Merge basket co-occurrence and text embeddings with PCA fusion.

    Parameters
    ----------
    basket_emb_df : pd.DataFrame
        Contains stockcode and basket_embedding columns
    text_emb_df : pd.DataFrame
        Contains stockcode and text_embedding columns
    alpha : float, default=1.0
        Weight for text embeddings
    beta : float, default=1.0
        Weight for basket embeddings
    final_dim : int, default=128
        Final embedding dimensionality

    Returns
    -------
    pd.DataFrame
        DataFrame with stockcode and embedding columns
    """
    # Merge both embedding types
    combined = text_emb_df.merge(
        basket_emb_df,
        on='stockcode',
        how='left'
    )

    # Fill missing basket embeddings with zeros
    missing_mask = combined['basket_embedding'].isna()
    if missing_mask.any():
        print(f"Items without basket data: {missing_mask.sum()}")
        combined.loc[missing_mask, 'basket_embedding'] = [
            np.zeros(basket_emb_df['basket_embedding'].iloc[0].shape)
            for _ in range(missing_mask.sum())
        ]

    # Stack embeddings horizontally
    X_text = np.vstack(combined['text_embedding'].values)
    X_basket = np.vstack(combined['basket_embedding'].values)

    # Apply weights
    X_text_weighted = X_text * alpha
    X_basket_weighted = X_basket * beta
    X_combined = np.hstack([X_text_weighted, X_basket_weighted])

    print(f"Combined shape before PCA: {X_combined.shape}")

    # Normalize
    X_combined = X_combined / (np.linalg.norm(X_combined, axis=1, keepdims=True) + 1e-9)

    # Apply PCA to reduce to final dimensionality
    pca = PCA(n_components=final_dim, random_state=42)
    X_final = pca.fit_transform(X_combined)

    # Final normalization
    X_final = X_final / (np.linalg.norm(X_final, axis=1, keepdims=True) + 1e-9)

    print(f"Final shape: {X_final.shape}")
    print(f"Variance explained: {pca.explained_variance_ratio_.sum()*100:.1f}%")

    # Create output
    result = pd.DataFrame({
        'stockcode': combined['stockcode'].astype(str),
        'embedding': list(X_final)
    })

    return result


def build_hybrid_embeddings(baskets, product_text_df, alpha=0.6, beta=0.6,
                            embedding_dim=128, sppmi_k_shift=5,
                            sbert_model='all-MiniLM-L6-v2'):
    """
    End-to-end hybrid embedding pipeline.

    Combines behavioral (SPPMI) and semantic (SBERT) embeddings.

    Parameters
    ----------
    baskets : list of lists
        Transaction baskets for co-occurrence analysis
    product_text_df : pd.DataFrame
        Product descriptions for semantic analysis
    alpha : float, default=0.6
        Weight for text embeddings
    beta : float, default=0.6
        Weight for basket embeddings
    embedding_dim : int, default=128
        Final embedding dimensionality
    sppmi_k_shift : float, default=5
        SPPMI shift parameter
    sbert_model : str, default='all-MiniLM-L6-v2'
        SBERT model name

    Returns
    -------
    tuple
        (item_embeddings_df, basket_embeddings_df, text_embeddings_df)
    """
    print("=" * 80)
    print("BUILDING HYBRID EMBEDDINGS")
    print("=" * 80)

    # Build basket embeddings
    print("\n1. Building basket embeddings (SPPMI)...")
    basket_embeddings = build_sppmi_embeddings(
        baskets,
        dim=embedding_dim,
        k_shift=sppmi_k_shift
    )

    # Build text embeddings
    print("\n2. Building text embeddings (SBERT)...")
    text_embeddings = build_text_embeddings(
        product_text_df,
        model_name=sbert_model
    )

    # Combine embeddings
    print("\n3. Combining embeddings with PCA fusion...")
    item_embeddings = combine_embeddings(
        basket_embeddings,
        text_embeddings,
        alpha=alpha,
        beta=beta,
        final_dim=embedding_dim
    )

    print("\n" + "=" * 80)
    print(f"Hybrid embeddings created successfully!")
    print(f"Total items: {len(item_embeddings):,}")
    print(f"Embedding dimension: {embedding_dim}")
    print("=" * 80)

    return item_embeddings, basket_embeddings, text_embeddings
