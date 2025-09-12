import pandas as pd

def detect_threshold_anomalies(df, col="heart_rate", threshold=150):
    return df[df[col] > threshold]

def detect_residual_anomalies(df, residual_col="residual", threshold=2):
    return df[df[residual_col].abs() > threshold]
