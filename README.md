# Time series prediction models
Reference of Time Series Modeling

# Models 
This task is a time series forecasting problem where the goal is to predict the volume in the next hour for each currency pair, based on features like past volumes, mid prices, and time intervals. Since the performance will be measured using Mean Absolute Percentage Error (MAPE), accuracy in predicting percentage changes is important.

Here are several approaches you can consider for building a model:

## 1. Classical Time Series Models (ARIMA, SARIMA)
Autoregressive Integrated Moving Average (ARIMA): ARIMA models are commonly used for univariate time series forecasting. They capture trends and seasonality well and can handle temporal dependencies. However, ARIMA does not handle multiple features (like mid-price) well.
Seasonal ARIMA (SARIMA): If there is a strong seasonality component (e.g., daily or weekly trading volume patterns), SARIMA can model it explicitly.
Limitations: ARIMA and SARIMA primarily work for univariate data and won’t utilize the full power of features like mid-price and last-hour volumes.

## 2. Regression-Based Models
Linear Regression: You can start with a simple linear regression model using your features (e.g., volume in the last minute, volume in the last hour, and mid-price) as predictors for the volume in the next hour.
Ridge or Lasso Regression: Regularized versions of linear regression can help if some features are not very predictive or if you want to avoid overfitting.
Strength: These models can be simple to implement and interpret.

Limitations: Linear models might not capture complex non-linear relationships in your data.

## 3. Tree-Based Models
Random Forest or Gradient Boosting (XGBoost, LightGBM): Tree-based ensemble methods are powerful for handling structured tabular data. These models can capture non-linear relationships and interactions between variables, making them a strong baseline for time series forecasting with multiple features.
Strengths:

Handle multiple features well (e.g., last-minute volume, last-hour volume, mid-price).
Can capture non-linearities and interactions in the data.
Require less pre-processing of data.
Limitations: They do not naturally handle temporal sequences, so you may need to carefully create lag features (e.g., previous hour volume, previous day volume) or use time-based cross-validation.

## 4. Deep Learning Models
Recurrent Neural Networks (RNNs) / LSTM / GRU: RNNs, especially Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models, are designed for sequential data and can capture long-term dependencies. These models are often used in time series forecasting tasks.
Temporal Convolutional Networks (TCNs): TCNs are convolutional neural networks adapted for sequence modeling. They can outperform RNN-based models in many time series forecasting tasks.
Strengths:

Capture long-term dependencies between observations in a sequence.
Can handle non-linearity, seasonality, and multiple features simultaneously.
Limitations:

Require larger amounts of data and careful tuning.
Harder to interpret compared to simpler models like linear regression or tree-based methods.

## 5. Hybrid Models
Auto-Regressive + Machine Learning: You can combine traditional time series models (like ARIMA) with machine learning models. For example, use ARIMA to handle the time component (e.g., seasonality, trends) and feed residuals (what ARIMA fails to predict) into a machine learning model like XGBoost to capture the rest of the signal.
6. Feature Engineering
Regardless of the model, here are a few feature engineering steps to improve performance:

Lag features: Include lagged versions of the volume and mid-price. For example, volume in the last minute, 5 minutes ago, 10 minutes ago, etc.
Rolling averages: Compute rolling averages or sums for the volume over various windows (e.g., last 10 minutes, last 30 minutes, last hour).
Time-based features: Include features like the hour of the day, day of the week, and whether the market is open or closed.
Seasonality: Look for patterns at daily, weekly, or monthly intervals and encode them as features.
Suggested Approach:
Given the problem involves multiple features (volumes and prices) and time dependencies, a tree-based ensemble model (like XGBoost or LightGBM) combined with careful feature engineering (lag features, rolling averages, etc.) would be a great starting point.

Tree-based models are excellent because they handle complex, non-linear relationships between variables without requiring too much data preprocessing.
Feature Engineering is key. Creating lags and rolling features will help the model understand temporal patterns.
If there’s strong seasonality, you could consider combining a simple time series model (e.g., ARIMA) with the residuals fed into a tree-based model.
You could further explore LSTM/GRU if you want to capture longer-term dependencies or expect the relationships to be heavily sequence-based.

Evaluation:
Since the performance is measured by MAPE (Mean Absolute Percentage Error), keep in mind that models should perform well at predicting relative percentage changes, rather than just absolute values.

Summary of the approach:
Baseline: Use a regression-based or tree-based model (like LightGBM) with features like past volume, mid-price, and engineered time-lag features.
Advanced: If baseline models don't perform well, explore RNN/LSTM or hybrid approaches to capture long-term dependencies.

