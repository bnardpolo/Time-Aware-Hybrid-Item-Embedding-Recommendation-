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
│   └── data.csv          # Raw UCI Online Retail dataset
├── artifacts/
│   ├── item_embeddings.pkl    # Final hybrid item vectors
│   ├── customer_vectors.pkl   # Final customer preference vectors
│   └── *.csv, *.png           # All optimization results and visualizations
└── clean_notebook (1).ipynb  # Main project notebook
```

### Dependencies

To run this notebook, install the following libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn tqdm
pip install sentence-transformers faiss-cpu # Use faiss-gpu if available
```
