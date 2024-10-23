"""
ARIMA Model:

For each currency pair, we fit an ARIMA model using ARIMA(volume_series, order=(p, d, q)). The (p, d, q) are the parameters for the ARIMA model:
p: The number of lag observations to include in the model (autoregressive component).
d: The degree of differencing (to make the series stationary).
q: The size of the moving average window (errors).
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_percentage_error
from itertools import product


fx_df = pd.DataFrame(data)
fx_df['datetime'] = pd.to_datetime(fx_df['datetime'])

# Shift the target variable to create 'volumeNextHour'
fx_df['volumeNextHour'] = fx_df.groupby('currency_pair')['volume_last_hour'].shift(-1)

# Drop rows with missing 'volumeNextHour' due to shifting
fx_df.dropna(subset=['volumeNextHour'], inplace=True)

# Function to train ARIMA model with exogenous variables for each currency pair
def train_arima_for_currency_pair(df, currency_pair, p=1, d=1, q=1):
    """
    Train an ARIMA model for a specific currency pair.
    
    Parameters:
    - df: DataFrame containing the FX data.
    - currency_pair: The currency pair to filter by (e.g., 'EURUSD').
    - p, d, q: ARIMA model parameters.
    
    Returns:
    - The fitted ARIMA model and the test dataset.
    """
    # Filter data for the specified currency pair
    df_pair = df[df['currency_pair'] == currency_pair].copy()
    
    # Set 'volume_last_hour' as the time series for ARIMA
    volume_series = df_pair['volume_last_hour']
    
    # Define exogenous variables: volume_last_minute, mid_price, spread
    exog_vars = df_pair[['volume_last_minute', 'mid_price', 'spread']]
    
    # Train ARIMA model with exogenous variables
    model = ARIMA(volume_series, exog=exog_vars, order=(p, d, q))
    model_fit = model.fit()
    
    return model_fit, df_pair

# Function to perform grid search for ARIMA parameters (p, d, q)
def optimize_arima(df, currency_pair, p_values, d_values, q_values):
    best_mape = float('inf')
    best_order = (0, 0, 0)
    best_model = None

    for p, d, q in product(p_values, d_values, q_values):
        try:
            # Train ARIMA model for the given order
            model_fit, df_pair = train_arima_for_currency_pair(df, currency_pair, p=p, d=d, q=q)
            
            # Prepare exogenous variables for prediction
            exog_test = df_pair[['volume_last_minute', 'mid_price', 'spread']].iloc[-1:].values

            # Forecast next hour volume
            forecast = model_fit.forecast(steps=1, exog=exog_test)[0]

            # Get actual volumeNextHour
            actual = df_pair['volumeNextHour'].iloc[-1]
            
            # Calculate MAPE
            mape = mean_absolute_percentage_error([actual], [forecast])

            # Update best model if MAPE is lower
            if mape < best_mape:
                best_mape = mape
                best_order = (p, d, q)
                best_model = model_fit

        except Exception as e:
            print(f"Error with ARIMA({p},{d},{q}) for {currency_pair}: {e}")
            continue

    return best_model, best_order, best_mape

# Define grid search ranges for ARIMA parameters
p_values = [0, 1, 2]
d_values = [0, 1]
q_values = [0, 1, 2]

# Prepare data for each currency pair and perform grid search for optimal ARIMA parameters
currency_pairs = fx_df['currency_pair'].unique()

best_models = {}
for pair in currency_pairs:
    best_model, best_order, best_mape = optimize_arima(fx_df, pair, p_values, d_values, q_values)
    best_models[pair] = (best_model, best_order, best_mape)
    print(f"Best ARIMA order for {pair}: {best_order} with MAPE: {best_mape}")

# Example output: Best model orders and MAPE for each currency pair
for pair, (model, order, mape) in best_models.items():
    print(f"Currency Pair: {pair}, Best ARIMA Order: {order}, MAPE: {mape}")


