"""Tests for preprocessing module."""

import pytest
import pandas as pd
import numpy as np
from src.preprocessing import (
    standardize_columns,
    fix_data_types,
    create_baskets,
    create_winsorized_features,
    create_log_features,
    optimize_dtypes
)


def test_standardize_columns():
    """Test column name standardization."""
    df = pd.DataFrame({
        'Invoice No': [1, 2],
        'Stock-Code': ['A', 'B'],
        'Unit Price': [1.0, 2.0]
    })

    result = standardize_columns(df)

    assert 'invoiceno' in result.columns
    assert 'stockcode' in result.columns
    assert 'unitprice' in result.columns
    assert len(result.columns) == 3


def test_fix_data_types():
    """Test data type conversion."""
    df = pd.DataFrame({
        'invoicedate': ['2011-01-01', '2011-01-02'],
        'quantity': ['1', '2'],
        'unitprice': ['10.5', '20.5'],
        'customerid': ['100', '200'],
        'stockcode': [' A001 ', ' B002 '],
        'description': ['Product A', 'Product B']
    })

    result = fix_data_types(df)

    assert pd.api.types.is_datetime64_any_dtype(result['invoicedate'])
    assert pd.api.types.is_numeric_dtype(result['quantity'])
    assert pd.api.types.is_numeric_dtype(result['unitprice'])
    assert result['stockcode'].iloc[0] == 'A001'  # Stripped
    assert result['description'].iloc[0] == 'product a'  # Lowercased


def test_create_baskets(sample_transactions):
    """Test basket creation from transactions."""
    baskets = create_baskets(sample_transactions)

    assert isinstance(baskets, list)
    assert len(baskets) == 3  # 3 unique invoices
    assert all(isinstance(basket, list) for basket in baskets)


def test_create_winsorized_features(sample_transactions):
    """Test winsorization of numeric features."""
    result = create_winsorized_features(sample_transactions)

    assert 'quantity_w' in result.columns
    assert 'unitprice_w' in result.columns
    assert 'revenue_w' in result.columns

    # Winsorized values should be within original range
    assert result['quantity_w'].min() >= result['quantity'].min()
    assert result['quantity_w'].max() <= result['quantity'].max()


def test_create_log_features(sample_transactions):
    """Test log transformation of features."""
    result = create_log_features(sample_transactions)

    assert 'log_quantity' in result.columns
    assert 'log_unitprice' in result.columns
    assert 'log_revenue' in result.columns

    # Log values should be non-negative
    assert (result['log_quantity'] >= 0).all()
    assert (result['log_unitprice'] >= 0).all()


def test_optimize_dtypes(sample_transactions):
    """Test memory optimization."""
    # Add int64 column
    df = sample_transactions.copy()
    df['test_int'] = df['customerid'].astype('int64')
    df['test_float'] = df['unitprice'].astype('float64')

    memory_before = df.memory_usage(deep=True).sum()
    result = optimize_dtypes(df)
    memory_after = result.memory_usage(deep=True).sum()

    assert memory_after <= memory_before
    # Check type downcasting
    assert result['test_int'].dtype == 'int32'
    assert result['test_float'].dtype == 'float32'
