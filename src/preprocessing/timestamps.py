import pandas as pd

def clean_timestamps(df: pd.DataFrame, time_col="timestamp") -> pd.DataFrame:
    """Convert to datetime and sort."""
    df[time_col] = pd.to_datetime(df[time_col])
    return df.sort_values(by=time_col).reset_index(drop=True)

def align_time_intervals(df: pd.DataFrame, time_col="timestamp", freq="1min") -> pd.DataFrame:
    """Reindex with fixed interval."""
    df = df.set_index(time_col).resample(freq).mean().interpolate()
    return df.reset_index()
