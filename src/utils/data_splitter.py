#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for splitting time series data.
"""

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger('cancellation_forecasting.data_splitter')

def split_data_by_year(df, train_expr='<=2023', test_expr='2024'):
    """
    Split DataFrame into training and validation sets based on year expressions.
    
    Args:
        df (pandas.DataFrame): DataFrame to split.
        train_expr (str): Expression for training data years (e.g., '<=2023', '2020-2022').
        test_expr (str): Expression for test data years (e.g., '2024', '>=2023').
        
    Returns:
        tuple: (training_df, validation_df, date_column_name)
    """
    # Find date column
    date_col = next((col for col in df.columns if pd.api.types.is_datetime64_dtype(df[col])), None)
    if not date_col:
        raise ValueError("No datetime column found in the DataFrame")
    
    # Add year column if it doesn't exist
    if 'Year' not in df.columns:
        df['Year'] = df[date_col].dt.year
    
    # Parse train_expr
    train_years = _parse_year_expression(train_expr)
    test_years = _parse_year_expression(test_expr)
    
    # Create filter for training and test data
    train_filter = np.zeros(len(df), dtype=bool)
    test_filter = np.zeros(len(df), dtype=bool)
    
    for year_filter in train_years:
        if isinstance(year_filter, tuple):
            op, value = year_filter
            if op == '>=':
                train_filter |= (df['Year'] >= value)
            elif op == '<=':
                train_filter |= (df['Year'] <= value)
            elif op == '>':
                train_filter |= (df['Year'] > value)
            elif op == '<':
                train_filter |= (df['Year'] < value)
        else:
            train_filter |= (df['Year'] == year_filter)
    
    for year_filter in test_years:
        if isinstance(year_filter, tuple):
            op, value = year_filter
            if op == '>=':
                test_filter |= (df['Year'] >= value)
            elif op == '<=':
                test_filter |= (df['Year'] <= value)
            elif op == '>':
                test_filter |= (df['Year'] > value)
            elif op == '<':
                test_filter |= (df['Year'] < value)
        else:
            test_filter |= (df['Year'] == year_filter)
    
    # Extract training and test sets
    df_train = df[train_filter].copy()
    df_test = df[test_filter].copy()
    
    # Log split information
    logger.info("Training data: %d records from %s to %s", 
               len(df_train),
               df_train[date_col].min().strftime('%Y-%m-%d'),
               df_train[date_col].max().strftime('%Y-%m-%d'))
    
    logger.info("Test data: %d records from %s to %s", 
               len(df_test),
               df_test[date_col].min().strftime('%Y-%m-%d') if not df_test.empty else 'N/A',
               df_test[date_col].max().strftime('%Y-%m-%d') if not df_test.empty else 'N/A')
    
    return df_train, df_test, date_col

def _parse_year_expression(expr):
    """
    Parse a year expression string into a list of filters.
    
    Args:
        expr (str): Year expression (e.g., '<=2023', '2020-2022', '2019,2021,2023')
        
    Returns:
        list: List of year values or (operator, value) tuples
    """
    filters = []
    
    # Handle comma-separated values
    for part in expr.split(','):
        part = part.strip()
        
        # Handle range (e.g., '2020-2022')
        if '-' in part and not part.startswith(('>=', '<=', '>', '<')):
            start, end = map(int, part.split('-'))
            for year in range(start, end + 1):
                filters.append(year)
        
        # Handle operators (e.g., '<=2023')
        elif part.startswith(('>=', '<=', '>', '<')):
            if part.startswith('>='):
                op = '>='
                value = int(part[2:])
            elif part.startswith('<='):
                op = '<='
                value = int(part[2:])
            elif part.startswith('>'):
                op = '>'
                value = int(part[1:])
            elif part.startswith('<'):
                op = '<'
                value = int(part[1:])
            
            filters.append((op, value))
        
        # Handle single year
        else:
            filters.append(int(part))
    
    return filters