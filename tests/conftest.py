"""Pytest configuration and shared fixtures."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


@pytest.fixture
def sample_transactions():
    """Create sample transaction data for testing."""
    np.random.seed(42)

    data = {
        'invoiceno': ['INV001'] * 5 + ['INV002'] * 3 + ['INV003'] * 4,
        'stockcode': ['A001', 'A002', 'A003', 'A004', 'A005',
                     'A001', 'A003', 'A006',
                     'A002', 'A004', 'A005', 'A007'],
        'description': ['product a', 'product b', 'product c', 'product d', 'product e',
                       'product a', 'product c', 'product f',
                       'product b', 'product d', 'product e', 'product g'],
        'quantity': [2, 1, 3, 1, 2, 1, 2, 1, 3, 1, 1, 2],
        'invoicedate': [datetime(2011, 1, 1)] * 5 + [datetime(2011, 2, 1)] * 3 + [datetime(2011, 3, 1)] * 4,
        'unitprice': [10.0, 20.0, 15.0, 25.0, 12.0, 10.0, 15.0, 30.0, 20.0, 25.0, 12.0, 18.0],
        'customerid': [1001] * 5 + [1002] * 3 + [1001] * 4,
        'country': ['UK'] * 12
    }

    df = pd.DataFrame(data)
    df['revenue'] = df['quantity'] * df['unitprice']

    return df


@pytest.fixture
def sample_baskets():
    """Create sample basket data for testing."""
    return [
        ['A001', 'A002', 'A003'],
        ['A001', 'A004'],
        ['A002', 'A003', 'A004'],
        ['A001', 'A002', 'A005'],
        ['A003', 'A004', 'A005']
    ]


@pytest.fixture
def sample_product_text():
    """Create sample product text data for testing."""
    data = {
        'stockcode': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'description': [
            'red apple fruit',
            'green banana fruit',
            'orange citrus fruit',
            'yellow lemon citrus',
            'purple grape fruit'
        ]
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_embeddings():
    """Create sample embeddings for testing."""
    np.random.seed(42)

    embeddings = np.random.randn(5, 128)
    # Normalize
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    data = {
        'stockcode': ['A001', 'A002', 'A003', 'A004', 'A005'],
        'embedding': list(embeddings)
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_ground_truth():
    """Create sample ground truth for evaluation."""
    return {
        1001: {'A001', 'A002', 'A003'},
        1002: {'A004', 'A005'}
    }
