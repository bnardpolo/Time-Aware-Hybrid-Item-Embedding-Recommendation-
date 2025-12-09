## Time-Aware Hybrid Item Embedding Recommendation: Fusion of Behavioral (SPPMI) and Semantic (SBERT) Signals

## Project Overview

This project implements a comprehensive, production-ready **Hybrid Item-Based Collaborative Filtering Recommender** for retail transaction data. The core innovation is the creation of a powerful item embedding vector that captures both:

1.  **Behavioral Patterns:** What items are purchased together (co-occurrence).
2.  **Semantic Similarity:** What items are described similarly (text analysis).

The system generates personalized recommendations by combining these signals and applying a **time-based exponential decay function** to accurately model current customer preferences.

## Core Methodologies

The notebook demonstrates proficiency across the full data science lifecycle:

| Component | Key Techniques | Description |
| :--- | :--- | :--- |
| **Data Engineering** | Memory Optimization, Winsorization, Log-Transformation | Efficient data handling including a **45% memory reduction** and robust outlier management. |
| **Behavioral Embeddings** | **SPPMI** (Shifted Positive Pointwise Mutual Information) | Learns item vectors based on co-purchase patterns derived from basket data. |
| **Semantic Embeddings** | **SBERT** (Sentence-BERT) / TF-IDF | Generates contextual vectors from product descriptions to capture semantic similarity. |
| **Hybrid Fusion** | **PCA** (Principal Component Analysis) Fusion | Merges the behavioral and semantic vectors into a single, optimized 128-dimensional item embedding. |
| **Customer Profiling** | **Recency Weighting** (60-day Half-life) | Aggregates purchased item embeddings, weighting recent purchases more heavily to reflect evolving customer intent. |
| **Scalable Retrieval** | **FAISS** (Facebook AI Similarity Search) | Builds an index for sub-millisecond nearest-neighbor search to ensure fast, real-time recommendation retrieval. |
| **Evaluation** | **Time-Based Split, Grid Search, NDCG@K** | Rigorous validation using a time-based test set and systematic weight optimization (alpha, beta). |

## Key Results

The grid search confirms that the optimized hybrid model significantly outperforms models relying solely on one data source.

| Model | Alpha (Text Weight) | Beta (Basket Weight) | Precision@10 | NDCG@10 |
| :--- | :--- | :--- | :--- | :--- |
| **Optimal Hybrid** | **0.6** | **0.6** | **0.3425** | **0.4787** |
| Text Only Baseline | 1.0 | 0.0 | 0.2995 | 0.4420 |
| Basket Only Baseline| 0.0 | 1.0 | 0.1735 | 0.2446 |

**Improvement Over Baselines (NDCG@10):**

  * vs. Text Only: **+8.30%**
  * vs. Basket Only: **+95.79%**

-----

## Repository and Setup

### Project Structure

```
.
├── data/
│   └── data.csv                    # Raw UCI Online Retail dataset
├── artifacts/                       # Generated artifacts (embeddings, models, plots)
├── src/                            # Modular Python package
│   ├── __init__.py
│   ├── preprocessing.py            # Data cleaning & feature engineering
│   ├── embeddings.py               # SPPMI, SBERT, hybrid fusion
│   ├── models.py                   # Recommender model & FAISS indexing
│   ├── evaluation.py               # Metrics & grid search
│   └── utils.py                    # Visualization & helpers
├── tests/                          # Comprehensive test suite
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_preprocessing.py
│   ├── test_embeddings.py
│   ├── test_models.py
│   └── test_evaluation.py
├── clean_notebook (1).ipynb        # Original notebook
├── example_usage.py                # Example script using modules
├── requirements.txt                # Python dependencies
├── setup.py                        # Package installation
├── pytest.ini                      # Test configuration
└── .gitignore
```

## Quick Start

### Installation

1. **Clone the repository:**
```bash
git clone <repo-url>
cd Time-Aware-Hybrid-Item-Embedding-Recommendation-
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
# Or install as a package
pip install -e .
```

3. **Download data:**
Place the UCI Online Retail dataset in `data/data.csv`

### Using the Modular Package

```python
from src import (
    load_and_preprocess_data,
    create_baskets,
    extract_product_text,
    build_hybrid_embeddings,
    HybridRecommender
)

# Load and preprocess data
df = load_and_preprocess_data('data/data.csv', artifacts_dir='artifacts')

# Create inputs for embedding generation
baskets = create_baskets(df)
product_text = extract_product_text(df)

# Build hybrid embeddings
item_embeddings, _, _ = build_hybrid_embeddings(
    baskets=baskets,
    product_text_df=product_text,
    alpha=0.6,  # Text weight
    beta=0.6    # Basket weight
)

# Create and train recommender
recommender = HybridRecommender(item_embeddings)
recommender.fit(df, halflife_days=60)

# Generate recommendations
recommendations = recommender.recommend(customer_id=12345, n_recommendations=10)

# Save model
recommender.save('artifacts/recommender_model.pkl')

# Load model later
recommender = HybridRecommender.load('artifacts/recommender_model.pkl')
```

### Running the Example Script

```bash
python example_usage.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py
```

## API Reference

### Preprocessing (`src.preprocessing`)
- `load_and_preprocess_data()` - End-to-end preprocessing pipeline
- `create_baskets()` - Extract transaction baskets
- `extract_product_text()` - Get canonical product descriptions
- `create_time_split()` - Create time-based train/test split

### Embeddings (`src.embeddings`)
- `build_sppmi_embeddings()` - Generate behavioral embeddings
- `build_text_embeddings()` - Generate semantic embeddings (SBERT/TF-IDF)
- `combine_embeddings()` - Hybrid fusion with PCA
- `build_hybrid_embeddings()` - End-to-end embedding pipeline

### Models (`src.models`)
- `HybridRecommender` - Main recommendation engine
  - `.fit()` - Build customer vectors from transactions
  - `.recommend()` - Generate recommendations for a customer
  - `.get_similar_items()` - Find similar items
  - `.save()` / `.load()` - Persist and load models

### Evaluation (`src.evaluation`)
- `precision_at_k()`, `recall_at_k()`, `ndcg_at_k()` - Metrics
- `evaluate_model()` - Evaluate recommender on test set
- `grid_search_weights()` - Optimize alpha/beta weights
- `find_best_weights()` - Extract best configuration from grid search
