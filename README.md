# Cancellation Forecasting

This project provides tools for forecasting cancellation dates using multiple forecasting models including Prophet, SARIMA, and Holt-Winters. The analysis helps predict future cancellation patterns based on historical data.

## Features

- Data preprocessing and exploration
- Multiple forecasting models:
  - Facebook Prophet
  - SARIMA-like model
  - Advanced Seasonal model with weighted recent data
  - Holt-Winters Triple Exponential Smoothing
- Model comparison and evaluation
- Visualization of forecasts and errors

## Installation

```bash
# Clone the repository
git clone https://github.com/AlanNika/cancellation-forecasting.git
cd cancellation-forecasting

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

1. Place your CSV file named `Cancellation_dates.csv` in the data directory
2. Run the main analysis script:

```bash
python src/main.py
```

## File Structure

```
cancellation-forecasting/
├── data/                    # Data files
├── src/                     # Source code
│   ├── models/              # Forecasting models
│   ├── utils/               # Utility functions
│   ├── visualization/       # Visualization functions
│   └── main.py              # Main execution script
├── tests/                   # Unit tests
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## Model Descriptions

### Facebook Prophet
Handles trend and seasonality automatically, robust to missing data and outliers, and can incorporate holiday effects.

### SARIMA-like Model
Provides a strong seasonal component based on historical monthly averages, with a linear trend component based on year-over-year growth. Good performance with stable patterns.

### Advanced Seasonal Model
Weights recent data more heavily, is adaptive to recent trend changes, and captures evolving seasonality patterns.

### Holt-Winters Model
Uses a triple exponential smoothing approach with separate components for level, trend, and seasonality, and has adjustable smoothing parameters.

## License

MIT