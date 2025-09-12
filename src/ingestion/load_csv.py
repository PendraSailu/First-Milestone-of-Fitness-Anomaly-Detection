import pandas as pd
import os

def load_csv(path=None):
    if path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # project root
        path = os.path.join(base_dir, "data", "sample_heart_rate.csv")
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df
