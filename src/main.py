#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script for cancellation date forecasting.

This script performs time series forecasting on cancellation data using 
multiple forecasting models and evaluates their performance.
"""

import os
import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('cancellation_forecasting')

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.data_loader import load_and_preprocess_data
from src.utils.data_splitter import split_data_by_year
from src.models.prophet_model import train_prophet_model, make_prophet_forecast
from src.models.sarima_model import build_sarima_like_model
from src.models.advanced_seasonal_model import build_advanced_seasonal_model
from src.models.holt_winters_model import build_holt_winters_model
from src.utils.evaluation import calculate_error_metrics
from src.visualization.visualize import (
    plot_historical_data,
    plot_forecast_comparison,
    plot_individual_forecasts,
    plot_monthly_errors
)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cancellation Date Forecasting')
    parser.add_argument('--data', type=str, default='data/Cancellation_dates.csv',
                      help='Path to input CSV file (default: data/Cancellation_dates.csv)')
    parser.add_argument('--separator', type=str, default=';',
                      help='CSV separator (default: ;)')
    parser.add_argument('--output', type=str, default='outputs',
                      help='Output directory for results (default: outputs)')
    parser.add_argument('--train-years', type=str, default='<=2023',
                      help='Years to use for training (default: <=2023)')
    parser.add_argument('--test-years', type=str, default='2024',
                      help='Years to use for testing (default: 2024)')
    parser.add_argument('--skip-prophet', action='store_true',
                      help='Skip Prophet model (useful if Prophet is not installed)')
    
    return parser.parse_args()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # 1. Load and preprocess data
    logger.info("Loading and preprocessing data from %s", args.data)
    df = load_and_preprocess_data(args.data, separator=args.separator)
    
    # 2. Split data into training and validation
    logger.info("Splitting data into training and validation sets")
    df_train, df_val, date_col = split_data_by_year(df, train_expr=args.train_years, 
                                                  test_expr=args.test_years)
    
    # 3. Aggregate data by month
    logger.info("Aggregating data by month")
    # Training data aggregation
    train_monthly = df_train.groupby(pd.Grouper(key=date_col, freq='M')).size().reset_index(name='Cancellations')
    train_monthly.sort_values(date_col, inplace=True)
    train_monthly['Year'] = train_monthly[date_col].dt.year
    train_monthly['Month'] = train_monthly[date_col].dt.month

    # Validation data aggregation (actual 2024 values)
    val_monthly = df_val.groupby(pd.Grouper(key=date_col, freq='M')).size().reset_index(name='Cancellations')
    val_monthly.sort_values(date_col, inplace=True)
    val_monthly['Year'] = val_monthly[date_col].dt.year
    val_monthly['Month'] = val_monthly[date_col].dt.month
    
    # Ensure all months in validation year are included
    val_year = int(args.test_years)
    all_months = pd.DataFrame({
        'year': val_year,
        'month': range(1, 13)
    })
    all_months[date_col] = pd.to_datetime(all_months.apply(
        lambda row: f'{row["year"]}-{row["month"]:02d}-15', axis=1))

    existing_months = set(val_monthly[date_col].dt.month)
    for month in range(1, 13):
        if month not in existing_months:
            new_row = pd.DataFrame({
                date_col: [pd.Timestamp(year=val_year, month=month, day=15)],
                'Cancellations': [0],
                'Year': [val_year],
                'Month': [month]
            })
            val_monthly = pd.concat([val_monthly, new_row], ignore_index=True)

    val_monthly.sort_values(date_col, inplace=True)
    
    # 4. Visualize historical data
    logger.info("Visualizing historical data")
    plot_historical_data(train_monthly, date_col, save_path=f"{args.output}/historical_data.png")
    
    # 5. Train and forecast with different models
    logger.info("Training forecasting models and generating predictions")
    forecast_results = {}
    
    # Prophet model
    if not args.skip_prophet:
        try:
            logger.info("Training Prophet model")
            prophet_df = train_monthly.rename(columns={date_col: 'ds', 'Cancellations': 'y'})
            model_prophet = train_prophet_model(prophet_df)
            prophet_forecast = make_prophet_forecast(model_prophet, val_year)
            forecast_results['Prophet'] = prophet_forecast
            
            # Plot Prophet forecast components
            fig = model_prophet.plot_components(prophet_forecast)
            plt.tight_layout()
            plt.savefig(f"{args.output}/prophet_components.png")
            plt.close()
            
        except Exception as e:
            logger.error("Error in Prophet forecasting: %s", e)
            logger.warning("If Prophet isn't installed, run: pip install prophet")
    
    # SARIMA-like model
    logger.info("Training SARIMA-like model")
    sarima_forecast = build_sarima_like_model(train_monthly, date_col, val_year)
    forecast_results['SARIMA'] = sarima_forecast
    
    # Advanced Seasonal model
    logger.info("Training Advanced Seasonal model")
    advanced_forecast = build_advanced_seasonal_model(train_monthly, date_col, val_year)
    forecast_results['Advanced'] = advanced_forecast
    
    # Holt-Winters model
    logger.info("Training Holt-Winters model")
    holt_winters_forecast = build_holt_winters_model(train_monthly, date_col, val_year)
    forecast_results['Holt-Winters'] = holt_winters_forecast
    
    # 6. Prepare validation data
    actual = val_monthly.rename(columns={date_col: 'ds', 'Cancellations': 'actual'})
    actual = actual[['ds', 'actual']]
    
    # 7. Create comparison dataframe
    common_dates = pd.DataFrame({'ds': [pd.Timestamp(val_year, month, 15) for month in range(1, 13)]})
    forecast_comparison = pd.merge(common_dates, actual, on='ds', how='left')
    
    # Add each model's forecasts
    for model_name, forecast_df in forecast_results.items():
        model_data = forecast_df[['ds', 'yhat']].copy()
        model_data = model_data.rename(columns={'yhat': model_name.lower()})
        forecast_comparison = pd.merge(
            forecast_comparison,
            model_data,
            on='ds',
            how='left'
        )
    
    # 8. Calculate error metrics
    logger.info("Calculating error metrics")
    forecast_metrics = forecast_comparison.dropna(subset=['actual']).copy()
    metrics = calculate_error_metrics(forecast_metrics, model_names=list(forecast_results.keys()))
    
    # Display metrics
    print("\nModel Performance Metrics:")
    print(metrics.round(2))
    
    # Identify best model
    best_model = metrics.loc[metrics['MAE'].idxmin()]
    print(f"\nBest performing model based on MAE: {metrics['MAE'].idxmin()}")
    print(f"MAE: {best_model['MAE']:.2f}, MAPE: {best_model['MAPE']:.2f}%, RMSE: {best_model['RMSE']:.2f}")
    
    # Save metrics to CSV
    metrics.to_csv(f"{args.output}/model_performance_metrics.csv")
    
    # 9. Visualize forecasts
    logger.info("Visualizing forecasts")
    plot_forecast_comparison(forecast_comparison, save_path=f"{args.output}/forecast_comparison.png")
    plot_individual_forecasts(forecast_comparison, save_path=f"{args.output}/individual_forecasts.png")
    
    # 10. Monthly error analysis
    logger.info("Analyzing monthly errors")
    monthly_errors = forecast_metrics.copy()
    
    for model_name in forecast_results.keys():
        model_lower = model_name.lower()
        monthly_errors[f'{model_lower}_error'] = (monthly_errors['actual'] - monthly_errors[model_lower]).abs()
    
    monthly_errors['Month'] = monthly_errors['ds'].dt.strftime('%Y-%m')
    
    # Error columns
    error_cols = ['Month'] + [f'{model.lower()}_error' for model in forecast_results.keys()]
    monthly_error_table = monthly_errors[error_cols].copy().round(1)
    
    # Display monthly errors
    print("\nMonthly Absolute Errors:")
    print(monthly_error_table)
    
    # Save monthly errors to CSV
    monthly_error_table.to_csv(f"{args.output}/monthly_errors.csv", index=False)
    
    # Visualize monthly errors
    plot_monthly_errors(monthly_error_table, save_path=f"{args.output}/monthly_errors.png")
    
    # 11. Save forecast comparison
    comparison_table = forecast_comparison.copy()
    comparison_table['Month'] = comparison_table['ds'].dt.strftime('%Y-%m')
    comparison_columns = ['Month', 'actual'] + [model.lower() for model in forecast_results.keys()]
    comparison_table = comparison_table[comparison_columns].round(1)
    comparison_table.to_csv(f"{args.output}/forecast_comparison.csv", index=False)
    
    logger.info("Analysis complete. Results saved to %s", args.output)


if __name__ == "__main__":
    main()
