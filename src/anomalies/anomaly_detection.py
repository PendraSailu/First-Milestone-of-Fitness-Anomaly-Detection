import numpy as np
import pandas as pd

def rule_based_anomalies(df: pd.DataFrame, col="heart_rate", min_val=40, max_val=180):
    """Detect rule-based anomalies."""
    return df[(df[col] < min_val) | (df[col] > max_val)]

def residual_anomalies(df: pd.DataFrame, forecast: pd.DataFrame, threshold=20):
    """Detect anomalies based on residuals from Prophet forecast."""
    df = df.copy()
    df["yhat"] = forecast["yhat"].values
    df["residual"] = df["heart_rate"] - df["yhat"]
    return df[np.abs(df["residual"]) > threshold]

def cluster_anomalies(features, labels):
    """Points assigned to noise cluster (-1 in DBSCAN)."""
    return features[labels == -1]
