import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error

# Sample dataset
data = {
    'datetime': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00', 
                 '2024-01-01 10:03:00', '2024-01-01 10:04:00', '2024-01-01 10:05:00', '2024-01-01 10:06:00'],
    'currency_pair': ['EURUSD', 'GBPUSD', 'EURCHF', 'EURUSD', 'GBPUSD', 'EURCHF', 'EURUSD'],
    'volume_last_minute': [120.5, 110.0, 135.3, 125.4, 105.6, 130.0, 118.0],
    'volume_last_hour': [5000.0, 4800.0, 5200.0, 5100.0, 4900.0, 5300.0, 5150.0],
    'mid_price': [1.1050, 1.3075, 1.0812, 1.1060, 1.3080, 1.0820, 1.1070],
}
fx_df = pd.DataFrame(data)
fx_df['datetime'] = pd.to_datetime(fx_df['datetime'])

# Function to prepare data and train ARIMA model for each currency pair
def train_arima_for_currency_pair(df, currency_pair, p=2, d=1, q=2):
    """
    Train an ARIMA model for a specific currency pair.
    
    Parameters:
    - df: DataFrame containing the FX data.
    - currency_pair: The currency pair to filter by (e.g., 'EURUSD').
    - p, d, q: ARIMA model parameters.
    
    Returns:
    - The fitted ARIMA model.
    """
    # Filter data for the specified currency pair
    df_pair = df[df['currency_pair'] == currency_pair].copy()
    
    # Set 'volume_last_hour' as the time series for ARIMA
    volume_series = df_pair['volume_last_hour']
    
    # Train ARIMA model
    model = ARIMA(volume_series, order=(p, d, q))
    model_fit = model.fit()
    
    return model_fit, df_pair

# Prepare data for each currency pair and train ARIMA models
currency_pairs = fx_df['currency_pair'].unique()

models = {}
for pair in currency_pairs:
    model_fit, df_pair = train_arima_for_currency_pair(fx_df, pair)
    models[pair] = (model_fit, df_pair)

# Make predictions using ARIMA for each currency pair and calculate MAPE
def evaluate_arima_model(models):
    predictions = []
    actuals = []
    
    for pair, (model_fit, df_pair) in models.items():
        # Get the predicted volume for the next hour (forecast one step ahead)
        forecast = model_fit.forecast(steps=1)[0]
        actual = df_pair['volume_last_hour'].shift(-1).iloc[-1]  # Target: volumeNextHour
        
        # Store predictions and actuals
        predictions.append(forecast)
        actuals.append(actual)
        
        print(f"Currency Pair: {pair}")
        print(f"Predicted VolumeNextHour: {forecast}")
        print(f"Actual VolumeNextHour: {actual}")
        print()
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f"Mean Absolute Percentage Error (MAPE): {mape}")

# Evaluate the models and calculate MAPE
evaluate_arima_model(models)
