"""
Data preprocessing module for the recommendation system.

This module handles data loading, cleaning, feature engineering,
and transformation of raw transaction data.
"""

import numpy as np
import pandas as pd
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


def standardize_columns(df):
    """
    Convert column names to lowercase and remove special characters.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with raw column names

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized column names
    """
    df = df.copy()
    df.columns = [
        col.strip().replace(' ', '').replace('-', '').lower()
        for col in df.columns
    ]
    return df


def assess_data_quality(df):
    """
    Generate comprehensive data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to assess

    Returns
    -------
    pd.DataFrame
        Quality report with null counts, percentages, and unique values
    """
    total_rows = len(df)
    report = []

    for col in df.columns:
        s = df[col]
        report.append({
            'column': col,
            'dtype': str(s.dtype),
            'non_null': int(s.notna().sum()),
            'null_count': int(s.isna().sum()),
            'null_pct': round(s.isna().mean() * 100, 2),
            'unique_values': int(s.nunique(dropna=True)),
            'sample': ' | '.join(s.dropna().astype(str).head(3).tolist())
        })

    report_df = pd.DataFrame(report)
    report_df = report_df.sort_values('null_pct', ascending=False)

    return report_df


def fix_data_types(df):
    """
    Convert columns to appropriate data types.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with mixed types

    Returns
    -------
    pd.DataFrame
        Dataframe with corrected data types
    """
    df = df.copy()

    # Convert datetime
    if 'invoicedate' in df.columns:
        df['invoicedate'] = pd.to_datetime(df['invoicedate'], errors='coerce')

    # Convert numeric columns
    numeric_cols = ['quantity', 'unitprice', 'customerid']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean string columns
    string_cols = ['invoiceno', 'stockcode', 'country']
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Handle description specially
    if 'description' in df.columns:
        df['description'] = (df['description']
                            .fillna('no description')
                            .astype(str)
                            .str.strip()
                            .str.lower())

    return df


def fill_missing_customer_ids(df):
    """
    Fill missing customer IDs using invoice-level logic.

    If an invoice has some valid customer IDs, use the most common one.
    If an invoice has no customer IDs, assign a synthetic ID.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with potentially missing customer IDs

    Returns
    -------
    pd.DataFrame
        Dataframe with all customer IDs filled
    """
    df = df.copy()

    # Step 1: Fill within invoices where possible
    invoice_customer_mode = (df.groupby('invoiceno')['customerid']
                            .apply(lambda x: x.dropna().mode().iloc[0]
                                   if not x.dropna().empty else np.nan))

    df['customerid'] = df['customerid'].fillna(
        df['invoiceno'].map(invoice_customer_mode)
    )

    # Step 2: Assign synthetic IDs for remaining missing values
    still_missing = df['customerid'].isna()

    if still_missing.any():
        # Get invoices that still need IDs
        invoices_needing_ids = df.loc[still_missing, 'invoiceno'].unique()

        # Create synthetic IDs starting from max existing ID
        existing_max = df['customerid'].dropna().max()
        base_id = int(existing_max) if pd.notna(existing_max) else 10000

        # Map each invoice to a unique synthetic ID
        synthetic_mapping = {
            inv: base_id + i + 1
            for i, inv in enumerate(invoices_needing_ids)
        }

        # Fill missing values
        mask = df['customerid'].isna()
        df.loc[mask, 'customerid'] = (df.loc[mask, 'invoiceno']
                                      .map(synthetic_mapping))

    # Convert to integer
    df['customerid'] = df['customerid'].astype('int64')

    return df


def clean_transactions(df):
    """
    Apply business rules to filter valid transactions.

    - Remove cancelled orders (invoice numbers starting with 'C')
    - Keep only positive quantities and prices
    - Calculate revenue
    - Remove duplicates

    Parameters
    ----------
    df : pd.DataFrame
        Raw transaction dataframe

    Returns
    -------
    pd.DataFrame
        Cleaned transaction dataframe with revenue column
    """
    df = df.copy()
    initial_rows = len(df)

    # Remove cancelled orders (invoice numbers starting with 'C')
    df = df[~df['invoiceno'].str.startswith('C', na=False)]

    # Keep only positive quantities and prices
    df = df[(df['quantity'] > 0) & (df['unitprice'] > 0)]

    # Calculate revenue
    df['revenue'] = df['quantity'] * df['unitprice']

    # Remove duplicates
    df = df.drop_duplicates()

    final_rows = len(df)
    removed = initial_rows - final_rows

    print(f"Rows removed: {removed:,} ({removed/initial_rows*100:.1f}%)")
    print(f"Final dataset: {final_rows:,} rows")

    return df


def analyze_numeric_distributions(df, columns):
    """
    Generate statistical summary with key percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    columns : list
        List of numeric column names to analyze

    Returns
    -------
    pd.DataFrame
        Statistical summary with percentiles
    """
    summary = []
    for col in columns:
        if col not in df.columns:
            continue

        s = df[col].dropna()
        summary.append({
            'column': col,
            'count': len(s),
            'mean': s.mean(),
            'std': s.std(),
            'min': s.min(),
            'p1': s.quantile(0.01),
            'p5': s.quantile(0.05),
            'p25': s.quantile(0.25),
            'p50': s.quantile(0.50),
            'p75': s.quantile(0.75),
            'p95': s.quantile(0.95),
            'p99': s.quantile(0.99),
            'max': s.max()
        })

    return pd.DataFrame(summary).set_index('column')


def create_winsorized_features(df, lower_pct=0.005, upper_pct=0.995):
    """
    Create winsorized versions of numeric columns to handle outliers.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    lower_pct : float
        Lower percentile for clipping
    upper_pct : float
        Upper percentile for clipping

    Returns
    -------
    pd.DataFrame
        Dataframe with additional winsorized columns
    """
    df = df.copy()

    # Winsorize quantity
    q_lower = df['quantity'].quantile(lower_pct)
    q_upper = df['quantity'].quantile(upper_pct)
    df['quantity_w'] = df['quantity'].clip(q_lower, q_upper)

    # Winsorize unitprice
    p_lower = df['unitprice'].quantile(lower_pct)
    p_upper = df['unitprice'].quantile(upper_pct)
    df['unitprice_w'] = df['unitprice'].clip(p_lower, p_upper)

    # Winsorize revenue (slightly more aggressive on upper end)
    r_lower = df['revenue'].quantile(lower_pct)
    r_upper = df['revenue'].quantile(0.990)  # 99th percentile for revenue
    df['revenue_w'] = df['revenue'].clip(r_lower, r_upper)

    return df


def create_log_features(df):
    """
    Create log-transformed features for skewed distributions.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with numeric columns

    Returns
    -------
    pd.DataFrame
        Dataframe with additional log-transformed columns
    """
    df = df.copy()

    # Log1p transformation (handles zeros gracefully)
    df['log_quantity'] = np.log1p(df['quantity'])
    df['log_unitprice'] = np.log1p(df['unitprice'])
    df['log_revenue'] = np.log1p(df['revenue'])

    return df


def optimize_dtypes(df):
    """
    Downcast numeric types to reduce memory usage.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe

    Returns
    -------
    pd.DataFrame
        Memory-optimized dataframe
    """
    df = df.copy()

    # Downcast integers where safe
    int_cols = df.select_dtypes(include=['int64']).columns
    for col in int_cols:
        if df[col].min() >= 0 and df[col].max() < 2**31:
            df[col] = df[col].astype('int32')

    # Downcast floats
    float_cols = df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        df[col] = df[col].astype('float32')

    # Convert low-cardinality strings to category
    for col in ['country', 'stockcode']:
        if col in df.columns and df[col].nunique() < 10000:
            df[col] = df[col].astype('category')

    return df


def create_baskets(df):
    """
    Extract item sequences per invoice for co-occurrence analysis.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe with invoiceno and stockcode columns

    Returns
    -------
    list
        List of baskets, where each basket is a list of item codes
    """
    # Sort by customer and time to maintain sequence order
    df_sorted = df.sort_values(['customerid', 'invoicedate', 'invoiceno'])

    # Group items by invoice
    baskets = (df_sorted
               .groupby('invoiceno')['stockcode']
               .apply(lambda x: x.astype(str).tolist())
               .tolist())

    print(f"Total baskets: {len(baskets):,}")
    print(f"Avg items per basket: {np.mean([len(b) for b in baskets]):.1f}")

    return baskets


def extract_product_text(df):
    """
    Get canonical description for each product.

    Uses the most common description (mode) for each stock code.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe with stockcode and description columns

    Returns
    -------
    pd.DataFrame
        Dataframe with stockcode and canonical description
    """
    # Use mode (most common description) per stockcode
    product_text = (df.groupby('stockcode')['description']
                    .apply(lambda x: x.dropna().mode().iloc[0]
                           if not x.dropna().empty else 'no description')
                    .reset_index())

    product_text.columns = ['stockcode', 'description']
    product_text['stockcode'] = product_text['stockcode'].astype(str)

    print(f"Products with text: {len(product_text):,}")

    return product_text


def create_time_split(df, split_date='2011-09-01'):
    """
    Create time-based split for evaluation.

    Test set contains each customer's last purchase after split date.

    Parameters
    ----------
    df : pd.DataFrame
        Transaction dataframe
    split_date : str
        Date string to split on (format: 'YYYY-MM-DD')

    Returns
    -------
    pd.Series
        Series mapping customer_id to set of items in their last purchase
    """
    split_timestamp = pd.Timestamp(split_date)

    # Get transactions after split date
    test_df = df[df['invoicedate'] >= split_timestamp].copy()

    # For each customer, get their last invoice
    last_invoices = (test_df.sort_values('invoicedate')
                     .groupby('customerid')['invoiceno']
                     .last()
                     .unique())

    # Create ground truth: items in last invoice per customer
    test_items = (df[df['invoiceno'].isin(last_invoices)]
                  .groupby('customerid')['stockcode']
                  .apply(lambda x: set(x.astype(str).tolist())))

    print(f"Split date: {split_date}")
    print(f"Test invoices: {len(last_invoices):,}")
    print(f"Test customers: {len(test_items):,}")

    return test_items


def load_and_preprocess_data(data_path, artifacts_dir=None):
    """
    End-to-end data loading and preprocessing pipeline.

    Parameters
    ----------
    data_path : str or Path
        Path to raw data CSV file
    artifacts_dir : str or Path, optional
        Directory to save intermediate artifacts

    Returns
    -------
    pd.DataFrame
        Fully preprocessed transaction dataframe
    """
    # Load data
    try:
        df_raw = pd.read_csv(data_path)
    except UnicodeDecodeError:
        df_raw = pd.read_csv(data_path, encoding='ISO-8859-1')

    print(f"Raw data shape: {df_raw.shape}")

    # Standardize columns
    df = standardize_columns(df_raw)

    # Fix data types
    df = fix_data_types(df)

    # Fill missing customer IDs
    df = fill_missing_customer_ids(df)

    # Clean transactions
    df = clean_transactions(df)

    # Create features
    df = create_winsorized_features(df)
    df = create_log_features(df)

    # Optimize memory
    memory_before = df.memory_usage(deep=True).sum() / 1024**2
    df = optimize_dtypes(df)
    memory_after = df.memory_usage(deep=True).sum() / 1024**2

    print(f"Memory usage: {memory_before:.1f} MB -> {memory_after:.1f} MB")
    print(f"Reduction: {(1 - memory_after/memory_before)*100:.1f}%")

    # Save if artifacts directory provided
    if artifacts_dir is not None:
        artifacts_path = Path(artifacts_dir)
        artifacts_path.mkdir(exist_ok=True)
        output_path = artifacts_path / "transactions_clean.csv"
        df.to_csv(output_path, index=False)
        print(f"\nCleaned dataset saved to: {output_path}")

    return df
