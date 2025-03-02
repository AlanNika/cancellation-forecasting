#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data loading and preprocessing utilities.
"""

import logging
import pandas as pd

logger = logging.getLogger('cancellation_forecasting.data_loader')

def load_and_preprocess_data(filepath, separator=';'):
    """
    Load and preprocess cancellation data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file.
        separator (str): CSV delimiter character (default: ';').
        
    Returns:
        pandas.DataFrame: Preprocessed DataFrame.
    """
    try:
        # Attempt to load the CSV file with specified separator
        df = pd.read_csv(filepath, sep=separator)
        
        # If the first column has all column names, split it
        if len(df.columns) == 1 and separator in df.columns[0]:
            first_col = df.columns[0]
            df = pd.read_csv(filepath, sep=separator, header=None)
            column_names = first_col.split(separator)
            df.columns = column_names
            
        # Log basic dataset information
        logger.info("Dataset loaded successfully")
        logger.info("Dimensions: %d rows, %d columns", *df.shape)
        logger.info("Columns: %s", ', '.join(df.columns))
        
    except FileNotFoundError:
        logger.error("Error: '%s' not found. Please check the path.", filepath)
        raise
    except Exception as e:
        logger.error("Error loading data: %s", e)
        raise
    
    # Process date column
    date_col = process_date_column(df)
    
    return df

def process_date_column(df):
    """
    Find and process the date column in the DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame to process.
        
    Returns:
        str: Name of the date column.
    """
    # Look for date column
    date_col = next((col for col in df.columns if 'date' in col.lower() or 'Move-out' in col), None)
    if not date_col:
        logger.error("No date column found!")
        raise ValueError("No date column found!")
    
    logger.info("Found date column: %s", date_col)
    
    # Convert date with proper error handling
    try:
        df[date_col] = pd.to_datetime(df[date_col], format='%d.%m.%Y', errors='coerce')
        
        # Check for failed conversions
        failed_dates = df[date_col].isnull().sum()
        if failed_dates > 0:
            logger.warning("%d dates couldn't be converted. Check format.", failed_dates)
        
        logger.info("Date range: %s to %s", 
                   df[date_col].min().strftime('%Y-%m-%d'),
                   df[date_col].max().strftime('%Y-%m-%d'))
                   
    except Exception as e:
        logger.error("Error converting date column: %s", e)
        raise
    
    return date_col