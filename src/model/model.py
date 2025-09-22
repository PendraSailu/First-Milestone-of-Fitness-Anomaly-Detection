import pandas as pd
from prophet import Prophet
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def fit_prophet(df: pd.DataFrame, time_col="timestamp", value_col="heart_rate"):
    """Fit Prophet model and return forecast."""
    prophet_df = df[[time_col, value_col]].rename(columns={time_col: "ds", value_col: "y"})
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)
    forecast = model.predict(prophet_df)
    return model, forecast

def detect_anomalies(df, time_col="timestamp", value_col="heart_rate"):
    """
    Detect anomalies in a time series using Prophet.

    Args:
        df (pd.DataFrame): Input dataframe with timestamp and value columns.
        time_col (str): Name of the timestamp column.
        value_col (str): Name of the value column.

    Returns:
        pd.DataFrame: Original dataframe with 'yhat' (forecast) and 'residual' columns.
    """
    # Prepare dataframe for Prophet
    prophet_df = df[[time_col, value_col]].rename(columns={time_col: "ds", value_col: "y"})
    prophet_df['ds'] = pd.to_datetime(prophet_df['ds'])  # Ensure datetime type

    # Fit Prophet model
    model = Prophet(daily_seasonality=True)
    model.fit(prophet_df)

    # Make forecast
    forecast = model.predict(prophet_df)
    forecast['ds'] = pd.to_datetime(forecast['ds'])  # Ensure datetime type

    # Merge forecast with original data safely
    merged = pd.merge(prophet_df, forecast[['ds', 'yhat']], on='ds', how='left')

    # Compute residuals
    merged['residual'] = merged['y'] - merged['yhat']

    # Restore original column names for compatibility with existing code
    merged = merged.rename(columns={"ds": time_col, "y": value_col})

    return merged

def cluster_data(features, n_clusters=5):
    """Cluster features safely for any shape."""
    # Convert 1D Series/array to 2D
    if len(features.shape) == 1:
        features = features.values.reshape(-1, 1)

    # Prevent n_clusters > n_samples
    n_samples = features.shape[0]
    if n_samples < n_clusters:
        n_clusters = max(1, n_samples)

    # Clean data
    features = pd.DataFrame(features).replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(features_scaled)

    return labels
