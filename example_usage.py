"""
Example script demonstrating how to use the modular recommendation system.

This script shows the complete workflow from data loading to generating recommendations.
"""

from pathlib import Path
from src import (
    load_and_preprocess_data,
    create_baskets,
    extract_product_text,
    create_time_split,
    build_hybrid_embeddings,
    HybridRecommender,
    evaluate_model
)


def main():
    """Run the complete recommendation pipeline."""

    # Setup paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    ARTIFACTS_DIR = BASE_DIR / "artifacts"
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("HYBRID RECOMMENDATION SYSTEM - EXAMPLE USAGE")
    print("=" * 80)

    # Step 1: Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df_clean = load_and_preprocess_data(
        data_path=DATA_DIR / "data.csv",
        artifacts_dir=ARTIFACTS_DIR
    )

    # Step 2: Create baskets for SPPMI
    print("\n2. Creating transaction baskets...")
    baskets = create_baskets(df_clean)

    # Step 3: Extract product descriptions
    print("\n3. Extracting product descriptions...")
    product_text = extract_product_text(df_clean)

    # Step 4: Build hybrid embeddings
    print("\n4. Building hybrid embeddings...")
    item_embeddings, basket_emb, text_emb = build_hybrid_embeddings(
        baskets=baskets,
        product_text_df=product_text,
        alpha=0.6,  # Text weight
        beta=0.6,   # Basket weight
        embedding_dim=128,
        sppmi_k_shift=5,
        sbert_model='all-MiniLM-L6-v2'
    )

    # Save embeddings
    item_embeddings.to_pickle(ARTIFACTS_DIR / 'item_embeddings.pkl')
    print(f"\nItem embeddings saved to: {ARTIFACTS_DIR / 'item_embeddings.pkl'}")

    # Step 5: Create recommender and fit customer vectors
    print("\n5. Building recommendation model...")
    recommender = HybridRecommender(item_embeddings)
    recommender.fit(df_clean, halflife_days=60)

    # Save trained model
    recommender.save(ARTIFACTS_DIR / 'recommender_model.pkl')
    print(f"Model saved to: {ARTIFACTS_DIR / 'recommender_model.pkl'}")

    # Step 6: Generate recommendations for a sample customer
    print("\n6. Generating sample recommendations...")
    sample_customer = df_clean['customerid'].iloc[0]
    recommendations = recommender.recommend(sample_customer, n_recommendations=10)

    print(f"\nTop 10 recommendations for customer {sample_customer}:")
    for rank, item_id in enumerate(recommendations, 1):
        # Get product description
        prod_desc = product_text[product_text['stockcode'] == item_id]['description'].values
        desc = prod_desc[0] if len(prod_desc) > 0 else 'N/A'
        print(f"  {rank:2d}. {item_id:10s} - {desc}")

    # Step 7: Evaluate on test set
    print("\n7. Evaluating on test set...")
    test_ground_truth = create_time_split(df_clean, split_date='2011-09-01')

    # Sample 100 customers for quick evaluation
    sample_customers = list(test_ground_truth.keys())[:100]

    eval_results = evaluate_model(
        recommender,
        test_ground_truth,
        k_values=[5, 10, 20],
        customer_subset=sample_customers
    )

    print("\nEvaluation Results:")
    print(eval_results.to_string(index=False))

    # Step 8: Find similar items
    print("\n8. Finding similar items...")
    sample_item = recommendations[0]
    similar_items = recommender.get_similar_items(sample_item, n_similar=5)

    print(f"\nItems similar to {sample_item}:")
    for rank, item_id in enumerate(similar_items, 1):
        prod_desc = product_text[product_text['stockcode'] == item_id]['description'].values
        desc = prod_desc[0] if len(prod_desc) > 0 else 'N/A'
        print(f"  {rank}. {item_id:10s} - {desc}")

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nArtifacts saved to: {ARTIFACTS_DIR}")
    print("\nTo load the trained model later:")
    print(f"  from src import HybridRecommender")
    print(f"  recommender = HybridRecommender.load('{ARTIFACTS_DIR / 'recommender_model.pkl'}')")


if __name__ == "__main__":
    main()
