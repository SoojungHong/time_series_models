import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import tensorflow as tf
from tcn import TCN, tcn_full_summary  # Temporal Convolutional Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Step 1: Load and preprocess the data
# Assuming df is the DataFrame containing your dataset with columns
# datetime, currency_pair, volume_last_minute, volume_last_hour, mid_price_of_currency_pair_at_given_timestamp

# Parse datetime and filter for a specific currency pair (e.g., EUR/USD)
df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

# Filter the data for a specific currency pair, e.g., EUR/USD
currency_pair = 'EUR/USD'
df = df[df['currency_pair'] == currency_pair]

# Drop the 'currency_pair' column as it's no longer needed
df.drop(columns=['currency_pair'], inplace=True)

# Step 2: Feature Engineering
# The target is the trade volume in the next hour, so we shift the `volume_last_hour` column to create the target.
df['target'] = df['volume_last_hour'].shift(-60)  # Shift the target by 60 minutes

# Drop rows with NaNs in the target column (these will be at the end of the dataset)
df.dropna(inplace=True)

# Split features (X) and target (y)
X = df[['volume_last_minute', 'volume_last_hour', 'mid_price_of_currency_pair_at_given_timestamp']]
y = df['target']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Step 4: Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data to 3D shape (samples, timesteps, features)
# TCNs expect a 3D input: (batch_size, timesteps, features). We'll treat each row as a timestep with multiple features.
# We can add an extra dimension to X_train_scaled and X_test_scaled to represent the "timesteps" dimension.

X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Step 5: Define the TCN model
# Using the TCN layer from the `keras-tcn` library
model = Sequential([
    TCN(input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]),  # (timesteps, features)
        dilations=[1, 2, 4, 8],         # Dilations control how far back the model looks at past data
        kernel_size=2,                  # Size of the convolution kernel
        nb_filters=64,                  # Number of filters in convolutional layers
        dropout_rate=0.2,               # Dropout for regularization
        return_sequences=False),        # We don't need sequences for forecasting
    Dense(64, activation='relu'),       # Dense layer for further processing
    Dense(1)                            # Output layer for the predicted volume (regression task)
])

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Print a summary of the model
tcn_full_summary(model)

# Step 6: Train the model
history = model.fit(X_train_scaled, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Step 7: Evaluate the model
# Predict on the test set
y_pred = model.predict(X_test_scaled)

# Calculate MAPE (Mean Absolute Percentage Error)
mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'MAPE: {mape:.4f}')

# Step 8: Plot predictions vs actuals
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(y_test.values, label='Actual Volume')
plt.plot(y_pred, label='Predicted Volume', color='red')
plt.title('Predicted vs Actual Volume')
plt.legend()
plt.show()

